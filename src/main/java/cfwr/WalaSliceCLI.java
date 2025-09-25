package cfwr;

import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.util.*;
import java.util.regex.Pattern;
import java.util.Collections;

import com.ibm.wala.types.ClassLoaderReference;
import com.ibm.wala.classLoader.IBytecodeMethod;
import com.ibm.wala.classLoader.IMethod;
import com.ibm.wala.classLoader.SourceDirectoryTreeModule;

import com.ibm.wala.classLoader.Language;
import com.ibm.wala.ipa.callgraph.*;
import com.ibm.wala.ipa.callgraph.AnalysisCacheImpl;
import com.ibm.wala.ipa.callgraph.impl.AllApplicationEntrypoints;
import com.ibm.wala.ipa.callgraph.impl.Util;
import com.ibm.wala.ipa.callgraph.propagation.PointerAnalysis;
import com.ibm.wala.ipa.cha.ClassHierarchyFactory;
import com.ibm.wala.ipa.cha.IClassHierarchy;

import com.ibm.wala.cast.java.ipa.callgraph.JavaSourceAnalysisScope;

import com.ibm.wala.ipa.slicer.NormalStatement;
import com.ibm.wala.ipa.slicer.Slicer;
import com.ibm.wala.ipa.slicer.Statement;

import com.ibm.wala.shrike.shrikeCT.InvalidClassFileException;

import com.ibm.wala.ssa.IR;
import com.ibm.wala.ssa.SSAInstruction;
import com.ibm.wala.types.MethodReference;

import com.ibm.wala.cast.java.ipa.callgraph.JavaSourceAnalysisScope;

/**
 * WalaSliceCLI (source-mode):
 *   Build a call graph from Java source roots, pick a seed instruction near a file:line inside
 *   the target method/field, take a backward thin slice, and write a textual slice.
 *
 * Required args:
 *   --sourceRoots "<dir1><pathsep><dir2>..."    (path-separator delimited, e.g. ':' on Linux, ';' on Windows)
 *   --projectRoot "<abs path to project root>"  (used to resolve files for output/materialization)
 *   --targetFile  "<relative or absolute path to .java file>"
 *   --line        "<1-based line>"
 *   --output      "<abs output directory>"
 *   exactly one of:
 *     --targetMethod "pkg.Cls#name(T1,T2,...)"
 *     --targetField  "pkg.Cls#FIELD"
 *
 * Example:
 *   java -jar build/libs/wala-slicer-all.jar \
 *     --sourceRoots "/proj/src/main/java:/proj/src/test/java" \
 *     --projectRoot "/proj" \
 *     --targetFile  "src/main/java/com/example/Foo.java" \
 *     --line 123 \
 *     --targetMethod "com.example.Foo#bar(int,String)" \
 *     --output "/tmp/slices/Foo__bar"
 */
public class WalaSliceCLI {

  public static void main(String[] argv) throws Exception {
    Map<String, String> a = parseArgs(argv);

    // --- required args
    Path projectRoot = Paths.get(req(a, "projectRoot")).toAbsolutePath().normalize();
    String sourceRootsStr = req(a, "sourceRoots");
    String targetFileArg  = req(a, "targetFile");
    Path targetFile = absolutizeUnderRoot(projectRoot, targetFileArg).normalize();
    int line = Integer.parseInt(req(a, "line"));
    Path outDir = Paths.get(req(a, "output")).toAbsolutePath().normalize();
    Files.createDirectories(outDir);

    String targetMethod = a.get("targetMethod");
    String targetField  = a.get("targetField");
    if ((targetMethod == null) == (targetField == null)) {
      die("Specify exactly ONE of --targetMethod or --targetField");
    }

    // --- 1) Compile Java files first, then use bytecode analysis
    System.out.println("Compiling Java files...");
    compileJavaFiles(sourceRootsStr, projectRoot);
    
    // --- 2) Build BYTECODE analysis scope (more reliable than source analysis)
    com.ibm.wala.ipa.callgraph.AnalysisScope scope = com.ibm.wala.ipa.callgraph.AnalysisScope.createJavaAnalysisScope();
    for (String rootDir : sourceRootsStr.split(Pattern.quote(File.pathSeparator))) {
      if (rootDir == null || rootDir.isBlank()) continue;
      Path p = Paths.get(rootDir).toAbsolutePath().normalize();
      if (Files.isDirectory(p)) {
        scope.addToScope(ClassLoaderReference.Application, new SourceDirectoryTreeModule(p.toFile()));
      } else {
        System.err.println("[warn] source root not found: " + p);
      }
    }

    // --- 3) CHA + options + builder (0-1-CFA is a good default)
    IClassHierarchy cha = ClassHierarchyFactory.makeWithRoot(scope);

    // If your project has a main(), this will find them; otherwise consider AllApplicationEntrypoints
    Iterable<Entrypoint> entries = Util.makeMainEntrypoints(scope, cha);
    if (!entries.iterator().hasNext()) {
      // Fallback so we still analyze even without an explicit "main"
      entries = new AllApplicationEntrypoints(scope, cha);
    }

    AnalysisOptions options = new AnalysisOptions(scope, entries);
    AnalysisCache cache = new AnalysisCacheImpl();

    CallGraphBuilder<?> builder =
        Util.makeZeroOneCFABuilder(Language.JAVA, options, cache, cha, scope);

    CallGraph cg = builder.makeCallGraph(options, null);
    PointerAnalysis<?> pa = builder.getPointerAnalysis();

    // --- 3) Resolve the target member to candidate CG nodes
    MemberCriterion crit = (targetMethod != null)
        ? MemberCriterion.method(targetMethod)
        : MemberCriterion.field(targetField);

    Set<CGNode> candidates = findCandidateNodes(cg, crit);
    if (candidates.isEmpty()) {
      die("No CGNode matched target member: " + crit.classDotMember);
    }

    // --- 4) Pick a seed instruction near the requested file:line
    Statement seed = findSeedStatement(candidates, targetFile, line);
    if (seed == null) {
      die("Could not find a seed Statement at " + targetFile + ":" + line);
    }

    // --- 5) Slice (backward thin)
    Slicer.DataDependenceOptions dataOpts = Slicer.DataDependenceOptions.NO_BASE_PTRS;
    Slicer.ControlDependenceOptions ctrlOpts = Slicer.ControlDependenceOptions.NONE;

    Collection<Statement> slice = Slicer.computeBackwardSlice(seed, cg, pa, dataOpts, ctrlOpts);

    // --- 6) Materialize to the ACTUAL target file only (no guessing)  <<<<<<<<<<<<<<<<<<<<<<<<
    // NEW: compute the path of the target file relative to projectRoot
    Path targetRel = projectRoot.relativize(targetFile);
    // NEW: collect lines only for that targetRel file
    SliceMaterialized sm = materializeSliceToText(slice, targetRel);
    writeTrimmedFiles(sm, projectRoot, outDir);
    writeManifest(sm, outDir);

    System.out.println("WALA slice wrote " + sm.fileToLines.size() + " file(s) to " + outDir);
  }

  // ========== helpers: args / paths ==========

  static Map<String,String> parseArgs(String[] argv) {
    Map<String,String> m = new LinkedHashMap<>();
    for (int i=0; i<argv.length; i++) {
      String k = argv[i];
      if (!k.startsWith("--")) continue;
      String key = k.substring(2);
      String val = "true";
      if (i+1 < argv.length && !argv[i+1].startsWith("--")) {
        val = argv[++i];
      }
      if ((val.startsWith("\"") && val.endsWith("\"")) || (val.startsWith("'") && val.endsWith("'"))) {
        val = val.substring(1, val.length()-1);
      }
      m.put(key, val);
    }
    return m;
  }

  static String req(Map<String,String> m, String k) {
    String v = m.get(k);
    if (v == null) die("Missing required arg: --" + k);
    return v;
  }

  static void die(String msg) { System.err.println(msg); System.exit(2); }

  static Path absolutizeUnderRoot(Path root, String p) {
    Path q = Paths.get(p);
    if (!q.isAbsolute()) q = root.resolve(q);
    return q.normalize();
  }

  // ========== member matching ==========

  /** "pkg.Clazz#method(T1,T2)" OR "pkg.Clazz#field" */
  static final class MemberCriterion {
    final String classDotMember;
    final boolean isMethod;
    private MemberCriterion(String s, boolean isMethod) { this.classDotMember = s; this.isMethod = isMethod; }
    static MemberCriterion method(String m) { return new MemberCriterion(m, true); }
    static MemberCriterion field(String f)  { return new MemberCriterion(f, false); }
  }

  static Set<CGNode> findCandidateNodes(CallGraph cg, MemberCriterion mc) {
    Set<CGNode> out = new HashSet<>();
    String[] parts = mc.classDotMember.split("#", 2);
    if (parts.length != 2) return out;
    String clazz  = parts[0];
    String member = parts[1];

    for (CGNode n : cg) {
      MethodReference mr = n.getMethod().getReference();
      if (mr == null) continue;
      String cls = mr.getDeclaringClass().getName().toString()
          .replace('/', '.').replaceAll("^L", "").replaceAll(";$", "");
      if (!cls.equals(clazz)) continue;

      if (mc.isMethod) {
        String mname = mr.getName().toString();
        if (!member.startsWith(mname + "(")) continue; // loose; tighten by parsing params if needed
        out.add(n);
      } else {
        // field: accept nodes in that class; the seed selection (file+line) will narrow it
        out.add(n);
      }
    }
    return out;
  }

  // ========== seed picking ==========

  static Statement findSeedStatement(Set<CGNode> cand, Path expectedFile, int line) {
    Statement best = null;
    int bestDist = Integer.MAX_VALUE;
    String expectedSimple = expectedFile.getFileName().toString();

    for (CGNode n : cand) {
      IR ir = n.getIR();
      if (ir == null) continue;

      IMethod m = n.getMethod();
      if (!(m instanceof IBytecodeMethod)) continue;
      IBytecodeMethod bm = (IBytecodeMethod) m;

      SSAInstruction[] insns = ir.getInstructions();
      if (insns == null) continue;

      // Heuristic: only consider instructions whose simple source filename matches our target file
      String guessed = guessSourceSimpleName(m);
      if (guessed != null && !guessed.equals(expectedSimple)) continue;

      for (int i = 0; i < insns.length; i++) {
        if (insns[i] == null) continue;
        try {
          int bcIndex = bm.getBytecodeIndex(i);
          if (bcIndex < 0) continue;
          int srcLine = bm.getLineNumber(bcIndex);
          if (srcLine < 0) continue;
          
          int d = Math.abs(srcLine - line);
          if (d < bestDist) { bestDist = d; best = new NormalStatement(n, i); }
        } catch (InvalidClassFileException e) {
          continue;
        }
      }
    }
    return best;
  }

  static String guessSourceSimpleName(IMethod m) {
    String cn = m.getDeclaringClass().getName().toString();
    cn = cn.replace('/', '.').replaceAll("^L", "").replaceAll(";$", "");
    String simple = cn.substring(cn.lastIndexOf('.') + 1);
    return simple + ".java";
  }

  // ========== materialization ==========

  static final class SliceMaterialized {
    final Map<Path, SortedSet<Integer>> fileToLines = new LinkedHashMap<>();
    final List<String> manifest = new ArrayList<>();
  }

  // CHANGED: only collect lines for the actual --targetFile (passed as targetRel)
  static SliceMaterialized materializeSliceToText(Collection<Statement> slice, Path targetRel) {
    SliceMaterialized sm = new SliceMaterialized();
    SortedSet<Integer> keep = new TreeSet<>();

    for (Statement st : slice) {
      if (!(st instanceof NormalStatement)) continue;
      NormalStatement ns = (NormalStatement) st;
      CGNode n = ns.getNode();
      IR ir = n.getIR();
      if (ir == null) continue;

      IMethod m = n.getMethod();
      if (!(m instanceof IBytecodeMethod)) continue;
      IBytecodeMethod bm = (IBytecodeMethod) m;

      int idx = ns.getInstructionIndex();
      try {
        int bcIndex = bm.getBytecodeIndex(idx);
        if (bcIndex < 0) continue;
        int srcLine = bm.getLineNumber(bcIndex);
        if (srcLine < 0) continue;

        // ONLY record lines; file is fixed to targetRel
        keep.add(srcLine);
      } catch (InvalidClassFileException e) {
        continue;
      }
    }

    if (!keep.isEmpty()) {
      sm.fileToLines.put(targetRel, keep);
      for (int l : keep) {
        sm.manifest.add(targetRel + ":" + l + "  // sliced");
      }
    }
    return sm;
  }

  // (kept for future multi-file variants; currently unused)
  static Path guessSourcePathRelative(IMethod m) {
    String cn = m.getDeclaringClass().getName().toString();
    cn = cn.replace('/', '/').replaceAll("^L", "").replaceAll(";$", "");
    String outer = cn.replaceAll("\\$.*$", "");
    return Paths.get(outer + ".java");
  }

  static String prettyStmt(NormalStatement ns) {
    return ns.getNode().getMethod().getReference().toString() + " @" + ns.getInstructionIndex();
  }

  static void writeTrimmedFiles(SliceMaterialized sm, Path projectRoot, Path outDir) throws IOException {
    for (Map.Entry<Path, SortedSet<Integer>> e : sm.fileToLines.entrySet()) {
      Path rel = e.getKey();
      Path src = projectRoot.resolve(rel).normalize();
      if (!Files.exists(src)) continue;

      List<String> all = Files.readAllLines(src, StandardCharsets.UTF_8);
      SortedSet<Integer> keep = e.getValue();

      // Keep only sliced lines (for CFG consumption; not guaranteed compilable)
      List<String> trimmed = new ArrayList<>();
      for (int i = 1; i <= all.size(); i++) {
        if (keep.contains(i)) trimmed.add(all.get(i - 1));
      }

      Path dest = outDir.resolve(rel);
      Files.createDirectories(dest.getParent());
      Files.write(dest, trimmed, StandardCharsets.UTF_8);
    }
  }

  static void writeManifest(SliceMaterialized sm, Path outDir) throws IOException {
    Path man = outDir.resolve("slice.manifest.txt");
    Files.createDirectories(outDir);
    Files.write(man, sm.manifest, StandardCharsets.UTF_8);
  }

  static void compileJavaFiles(String sourceRootsStr, Path projectRoot) throws IOException {
    // Simple Java compilation using javac - only compile test files, skip annotated JDK
    for (String rootDir : sourceRootsStr.split(Pattern.quote(File.pathSeparator))) {
      if (rootDir == null || rootDir.isBlank()) continue;
      Path sourceDir = Paths.get(rootDir).toAbsolutePath().normalize();
      if (!Files.isDirectory(sourceDir)) continue;
      
      // Find Java files, but exclude annotated-jdk directory
      List<Path> javaFiles = new ArrayList<>();
      Files.walk(sourceDir)
        .filter(p -> p.toString().endsWith(".java"))
        .filter(p -> !p.toString().contains("annotated-jdk")) // Skip annotated JDK files
        .forEach(javaFiles::add);
      
      if (javaFiles.isEmpty()) continue;
      
      // Compile Java files
      ProcessBuilder pb = new ProcessBuilder("javac");
      pb.command().add("-cp");
      // Add runtime classpath plus Checker Framework jars
      String rtCp = System.getProperty("java.class.path");
      String cfDist = "/home/ubuntu/checker-framework-3.42.0/checker/dist";
      String cfCp = String.join(File.pathSeparator,
        cfDist + "/checker.jar",
        cfDist + "/checker-qual.jar",
        cfDist + "/plume-util.jar",
        cfDist + "/javaparser-core-3.26.2.jar"
      );
      pb.command().add(rtCp + File.pathSeparator + cfCp);
      pb.command().add("-d");
      pb.command().add(sourceDir.toString()); // Output to same directory
      
      // Add all Java files
      for (Path javaFile : javaFiles) {
        pb.command().add(javaFile.toString());
      }
      
      try {
        Process proc = pb.start();
        int exitCode = proc.waitFor();
        if (exitCode != 0) {
          System.err.println("Warning: Java compilation failed with exit code " + exitCode);
          // Read error output
          try (var reader = new java.io.BufferedReader(new java.io.InputStreamReader(proc.getErrorStream()))) {
            reader.lines().forEach(System.err::println);
          }
        } else {
          System.out.println("Successfully compiled " + javaFiles.size() + " Java files in " + sourceDir);
        }
      } catch (InterruptedException e) {
        Thread.currentThread().interrupt();
        throw new IOException("Compilation interrupted", e);
      }
    }
  }
}

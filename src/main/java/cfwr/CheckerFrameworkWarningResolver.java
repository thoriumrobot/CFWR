package cfwr;

import com.github.javaparser.JavaParser;
import com.github.javaparser.ParseResult;
import com.github.javaparser.Position;
import com.github.javaparser.Range;
import com.github.javaparser.ast.CompilationUnit;
import com.github.javaparser.ast.body.*;
import com.github.javaparser.ast.Node;
import com.github.javaparser.ast.body.VariableDeclarator; // Corrected import

import java.io.*;
import java.nio.file.*;
import java.util.*;
import java.util.regex.Pattern;
import java.util.regex.Matcher;

/**
 * A utility program that processes a list of warnings from the Checker Framework
 * generated on a target Java program and outputs the field or method signatures
 * of the nearest enclosing field/method for each warning location.
 *
 * The purpose of the program is to take a list of warnings and then use Specimin
 * to generate a slice for each warning in the input list.
 *
 * Usage:
 * ./gradlew run -PappArgs="<projectRoot> <warningsFilePath> <resolverRoot>"
 *
 * Arguments:
 * - projectRoot: Absolute path to the root directory of the target Java project.
 * - warningsFilePath: Absolute path to the file containing the Checker Framework warnings.
 * - resolverRoot: Absolute path to the root directory of this tool (CFWR).
 *
 * The warnings file should contain warnings in the standard format output by the Checker Framework.
 *
 * Example warning format:
 * /path/to/File.java:25:17: compiler.err.proc.messager: [index] Possible out-of-bounds access
 *
 * Example invocation:
 * ./gradlew run -PappArgs="/path/to/project /path/to/warnings.txt /path/to/CFWR"
 */
public class CheckerFrameworkWarningResolver {

    /**
     * A pattern that matches the format of the warnings produced by the Checker Framework.
     * It is intended to match lines like the following:
     * /path/to/File.java:25:17: compiler.err.proc.messager: [index] Possible out-of-bounds access
     */
    private static final Pattern WARNING_PATTERN = Pattern.compile("^(.+\\.java):(\\d+):(\\d+):\\s*(compiler\\.(warn|err)\\.proc\\.messager):\\s*\\[(.+?)\\]\\s*(.*)$");

    static String resolverPath;
    static boolean executeCommandFlag = true; // Flag to control command execution

    public static void main(String[] args) {
        if (args.length < 3) {
            System.err.println("Usage: java CheckerFrameworkWarningResolver <projectRoot> <warningsFilePath> <resolverRoot>");
            return;
        }

        String projectRoot = args[0];
        String warningsFilePath = args[1];
        resolverPath = args[2];

        try {
            JavaParser parser = new JavaParser();

            List<Warning> warnings = new ArrayList<>();

            try (BufferedReader br = Files.newBufferedReader(Paths.get(warningsFilePath))) {
                String line;
                while ((line = br.readLine()) != null) {
                    Matcher matcher = WARNING_PATTERN.matcher(line);
                    if (matcher.matches()) {
                        String fileName = matcher.group(1).trim();
                        int lineNumber = Integer.parseInt(matcher.group(2).trim());
                        int columnNumber = Integer.parseInt(matcher.group(3).trim());
                        String compilerMessageType = matcher.group(4).trim();
                        String checkerName = matcher.group(6).trim();
                        String message = matcher.group(7).trim();

                        Path filePath = Paths.get(fileName);
                        if (!filePath.isAbsolute()) {
                            filePath = Paths.get(projectRoot).resolve(filePath).normalize();
                        }

                        warnings.add(new Warning(filePath, lineNumber, columnNumber, compilerMessageType, checkerName, message));
                    } else {
                        System.err.println("Warning line does not match expected format: " + line);
                    }
                }
            }

            Set<Path> filesToParse = new HashSet<>();
            for (Warning warning : warnings) {
                filesToParse.add(warning.filePath);
            }

            Map<Path, CompilationUnit> compilationUnits = new HashMap<>();
            for (Path filePath : filesToParse) {
                try {
                    ParseResult<CompilationUnit> result = parser.parse(filePath);
                    if (result.isSuccessful() && result.getResult().isPresent()) {
                        compilationUnits.put(filePath, result.getResult().get());
                    } else {
                        System.err.println("Failed to parse file: " + filePath);
                    }
                } catch (IOException e) {
                    System.err.println("Error reading file: " + filePath);
                    e.printStackTrace();
                }
            }

            for (Warning warning : warnings) {
                processWarning(warning, compilationUnits, projectRoot);
            }

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private static void processWarning(Warning warning, Map<Path, CompilationUnit> compilationUnits, String projectRoot) {
        try {
            CompilationUnit compilationUnit = compilationUnits.get(warning.filePath);
            if (compilationUnit == null) {
                System.err.println("No compilation unit found for file: " + warning.filePath);
                return;
            }

            Position warningPosition = new Position(warning.lineNumber, warning.columnNumber);

            Optional<BodyDeclaration<?>> enclosingMember = findEnclosingMember(compilationUnit, warningPosition);

            if (enclosingMember.isPresent()) {
                List<String> command = buildSpeciminCommand(enclosingMember.get(), warning, projectRoot);
                if (command != null) {
                    System.out.println(String.join(" ", command));
                    if (executeCommandFlag) {
                        // Execute within the specimin subdirectory
                        executeCommand(command, Paths.get(resolverPath, "specimin").toString());
                    }
                }
            } else {
                System.err.println("No enclosing member found for warning at " + warning.filePath + ":" + warning.lineNumber + ":" + warning.columnNumber);
            }
        } catch (Exception e) {
            System.err.println("Error processing warning: " + warning);
            e.printStackTrace();
        }
    }

    private static Optional<BodyDeclaration<?>> findEnclosingMember(CompilationUnit cu, Position position) {
        // Adjusted the list to use raw type to avoid type inference issues
        List<BodyDeclaration> bodyDeclarations = cu.findAll(BodyDeclaration.class);

        BodyDeclaration<?> closestMember = null;
        int smallestRange = Integer.MAX_VALUE;

        for (BodyDeclaration<?> member : bodyDeclarations) {
            if (member.getBegin().isPresent() && member.getEnd().isPresent()) {
                Range range = new Range(member.getBegin().get(), member.getEnd().get());

                if (range.contains(position)) {
                    int rangeSize = range.getLineCount();

                    if (rangeSize < smallestRange) {
                        smallestRange = rangeSize;
                        closestMember = member;
                    }
                }
            }
        }

        return Optional.ofNullable(closestMember);
    }

    private static List<String> buildSpeciminCommand(BodyDeclaration<?> member, Warning warning, String projectRoot) throws IOException {
        String baseSlicesDir = System.getenv().getOrDefault("SLICES_DIR", "slices");
        Path baseDirPath = Paths.get(baseSlicesDir).toAbsolutePath().normalize();
        Files.createDirectories(baseDirPath);

        String root = projectRoot;
        String targetFile = warning.filePath.toString();
        // Specimin expects targetFile paths relative to --root. If absolute and under root, relativize it.
        try {
            Path rootPath = Paths.get(root).toAbsolutePath().normalize();
            Path targetPath = Paths.get(targetFile).toAbsolutePath().normalize();
            if (targetPath.startsWith(rootPath)) {
                targetFile = rootPath.relativize(targetPath).toString();
            }
        } catch (Exception ignored) { }
        String targetMethodOrField;

        String sliceNameComponent;
        if (member instanceof MethodDeclaration) {
            MethodDeclaration method = (MethodDeclaration) member;
            String qualifiedClassName = getQualifiedClassName(method);
            String methodSignature = getMethodSignature(method);
            targetMethodOrField = qualifiedClassName + "#" + methodSignature;
            sliceNameComponent = qualifiedClassName + "#" + methodSignature;
        } else if (member instanceof ConstructorDeclaration) {
            ConstructorDeclaration constructor = (ConstructorDeclaration) member;
            String qualifiedClassName = getQualifiedClassName(constructor);
            String methodSignature = getConstructorSignature(constructor);
            targetMethodOrField = qualifiedClassName + "#" + methodSignature;
            sliceNameComponent = qualifiedClassName + "#" + methodSignature;
        } else if (member instanceof FieldDeclaration) {
            FieldDeclaration field = (FieldDeclaration) member;
            VariableDeclarator variable = findVariableAtPosition(field, new Position(warning.lineNumber, warning.columnNumber));
            if (variable != null) {
                String qualifiedClassName = getQualifiedClassName(field);
                String fieldName = variable.getNameAsString();
                targetMethodOrField = qualifiedClassName + "#" + fieldName;
                sliceNameComponent = qualifiedClassName + "#" + fieldName;
            } else {
                System.err.println("No variable found at position in field declaration");
                return null;
            }
        } else {
            System.err.println("Unsupported member type: " + member.getClass().getSimpleName());
            return null;
        }

        String relativeTargetFile = targetFile;
        try {
            Path rootPath = Paths.get(root).toAbsolutePath().normalize();
            Path targetPath = Paths.get(targetFile).toAbsolutePath().normalize();
            if (targetPath.startsWith(rootPath)) {
                relativeTargetFile = rootPath.relativize(targetPath).toString();
            }
        } catch (Exception ignored) { }

        String safeSliceDirName = sanitizeSliceName(relativeTargetFile + "__" + sliceNameComponent);
        Path outputPath = baseDirPath.resolve(safeSliceDirName);
        Files.createDirectories(outputPath);
        String outputDirectory = outputPath.toString();

        List<String> command = new ArrayList<>();
        command.add("./gradlew");
        command.add("run");
        String args;
        if (member instanceof FieldDeclaration) {
            args = String.join(" ",
                    "--outputDirectory", "\"" + outputDirectory + "\"",
                    "--root", "\"" + root + "\"",
                    "--targetFile", "\"" + targetFile + "\"",
                    "--targetField", "\"" + targetMethodOrField + "\""
            );
        } else {
            args = String.join(" ",
                    "--outputDirectory", "\"" + outputDirectory + "\"",
                    "--root", "\"" + root + "\"",
                    "--targetFile", "\"" + targetFile + "\"",
                    "--targetMethod", "\"" + targetMethodOrField + "\""
            );
        }
        command.add("--args=" + args);

        return command;
    }

    private static String sanitizeSliceName(String name) {
        return name.replaceAll("[^A-Za-z0-9._-]", "_");
    }

    private static String getQualifiedClassName(Node node) {
        // Traverse up to find the enclosing ClassOrInterfaceDeclaration
        Optional<ClassOrInterfaceDeclaration> classDecl = node.findAncestor(ClassOrInterfaceDeclaration.class);
        if (classDecl.isPresent()) {
            String className = classDecl.get().getNameAsString();
            // Get package name
            Optional<CompilationUnit> cu = node.findCompilationUnit();
            String packageName = cu.flatMap(CompilationUnit::getPackageDeclaration)
                    .map(pd -> pd.getNameAsString())
                    .orElse("");
            if (!packageName.isEmpty()) {
                return packageName + "." + className;
            } else {
                return className;
            }
        } else {
            return ""; // Or handle anonymous classes if necessary
        }
    }

    private static String getMethodSignature(MethodDeclaration method) {
        StringBuilder signature = new StringBuilder();
        signature.append(method.getNameAsString());
        signature.append("(");
        List<String> params = new ArrayList<>();
        for (Parameter param : method.getParameters()) {
            params.add(param.getType().asString());
        }
        signature.append(String.join(",", params));
        signature.append(")");
        return signature.toString();
    }

    private static String getConstructorSignature(ConstructorDeclaration constructor) {
        StringBuilder signature = new StringBuilder();
        signature.append(constructor.getNameAsString());
        signature.append("(");
        List<String> params = new ArrayList<>();
        for (Parameter param : constructor.getParameters()) {
            params.add(param.getType().asString());
        }
        signature.append(String.join(",", params));
        signature.append(")");
        return signature.toString();
    }

    private static VariableDeclarator findVariableAtPosition(FieldDeclaration field, Position position) {
        for (VariableDeclarator variable : field.getVariables()) {
            if (variable.getBegin().isPresent() && variable.getEnd().isPresent()) {
                Range range = new Range(variable.getBegin().get(), variable.getEnd().get());
                if (range.contains(position)) {
                    return variable;
                }
            }
        }
        return null;
    }

    private static String getTempDir() throws IOException {
        Path tempDirectory = Files.createTempDirectory("cfwr_");
        return tempDirectory.toAbsolutePath().toString();
    }

    private static void executeCommand(List<String> command, String workingDirectory) {
        try {
            ProcessBuilder processBuilder = new ProcessBuilder(command);
            processBuilder.directory(new File(workingDirectory));
            processBuilder.redirectErrorStream(true);
            Process process = processBuilder.start();

            // Capture output
            BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()));
            String line;
            while((line = reader.readLine()) != null) {
                System.out.println(line);
            }

            int exitCode = process.waitFor();
            if (exitCode != 0) {
                System.err.println("Command exited with code " + exitCode);
            } else {
                System.out.println("Command executed successfully: " + String.join(" ", command));
            }
        } catch (IOException | InterruptedException e) {
            System.err.println("Error executing command: " + String.join(" ", command));
            e.printStackTrace();
        }
    }

    private static class Warning {
        Path filePath;
        int lineNumber;
        int columnNumber;
        String compilerMessageType; // 'compiler.warn.proc.messager' or 'compiler.err.proc.messager'
        String checkerName;
        String message;

        Warning(Path filePath, int lineNumber, int columnNumber, String compilerMessageType, String checkerName, String message) {
            this.filePath = filePath;
            this.lineNumber = lineNumber;
            this.columnNumber = columnNumber;
            this.compilerMessageType = compilerMessageType;
            this.checkerName = checkerName;
            this.message = message;
        }

        @Override
        public String toString() {
            return filePath + ":" + lineNumber + ":" + columnNumber + ": " + compilerMessageType + ": [" + checkerName + "] " + message;
        }
    }
}

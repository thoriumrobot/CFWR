// TODO: all classes should be in some package

import com.github.javaparser.JavaParser;
import com.github.javaparser.ast.CompilationUnit;
import com.github.javaparser.ast.body.BodyDeclaration;
import com.github.javaparser.ast.body.ClassOrInterfaceDeclaration;
import com.github.javaparser.ast.body.FieldDeclaration;
import com.github.javaparser.ast.body.MethodDeclaration;
import com.github.javaparser.ast.visitor.VoidVisitorAdapter;
import com.github.javaparser.ParseResult;
import com.github.javaparser.utils.SourceRoot;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

/**
 * A utility program that processes a list of warnings from the Checker Framework
 * that were generated on some target program and outputs (to the console) the field or method signature
 * of the nearest enclosing field/method for each warning location. The purpose of
 * the program is to take a list of warnings and then use Specimin to generate a slice
 * for each warning in the input list. Currently, the formatted Specimin output is shown.
 */

public class CheckerFrameworkWarningResolver {

    /**
     * A pattern that matches the format of the warnings produced by the Checker Framework.
     * It is intended to match lines like the following:
     * TODO: an example
     */
    private static final Pattern WARNING_PATTERN = Pattern.compile("^.*\\.java:\\d+:\\d+:.*$");
    /**
     * Primary entry point. Usage: java CheckerFrameworkWarningResolver <projectRoot> <warningsFilePath> <CFWRroot>.
     * projectRoot and warningsFilePath are assumed to be absolute paths to the directory or text file.
     *
     * TODO: document how the user should produce the warnings file.
     *
     * Example: ./gradlew run -PappArgs="/home/ubuntu/naenv/checker-framework/checker/tests/index/ /home/ubuntu/naenv/checker-framework/index1.out /home/ubuntu/naenv/CFWR/"
     * @param args command-line arguments
     */

    static String resolverPath;
    static boolean executeCommandFlag = true; // Add a flag to control command execution

    public static void main(String[] args) {
        if (args.length < 2) {
            System.err.println("Usage: java CheckerFrameworkWarningResolver <projectRoot> <warningsFilePath> <resolverRoot>");
            return;
        }

        String projectRoot = args[0];
        String warningsFilePath = args[1];
        resolverPath = args[2];

        try {
            JavaParser parser = new JavaParser();
            SourceRoot sourceRoot = new SourceRoot(Paths.get(projectRoot));
            List<ParseResult<CompilationUnit>> parseResults = sourceRoot.tryToParse("");

            List<String> warnings = Files.readAllLines(Paths.get(warningsFilePath)).stream()
                    .filter(line -> WARNING_PATTERN.matcher(line).matches())
                    .collect(Collectors.toList());

            for (String warning : warnings) {
                processWarning(warning, parseResults, projectRoot);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    /**
     * Processes a single warning and prints the nearest enclosing method/field signature to that warning.
     *
     * @param warning the warning string
     * @param parseResults the program that was being analyzed when the given warning was generated, parsed by JavaParser
     * @param projectRoot the root of the target program's source tree TODO: why is this necessary?
     */
     private static void processWarning(String warning, List<ParseResult<CompilationUnit>> parseResults, String projectRoot) {
        try {
            String[] parts = warning.split(":");
            String fileName = parts[0].trim();
            int lineNumber = Integer.parseInt(parts[1].trim());
            int columnNumber = Integer.parseInt(parts[2].trim());
            String warningType = parts[3].split(" ")[1].trim();

            Path filePath = Paths.get(projectRoot, fileName).normalize();

            for (ParseResult<CompilationUnit> parseResult : parseResults) {
                if (parseResult.isSuccessful() && parseResult.getResult().isPresent()) {
                    CompilationUnit compilationUnit = parseResult.getResult().get();
                    if (compilationUnit.getStorage().isPresent() &&
                            compilationUnit.getStorage().get().getPath().equals(filePath)) {
                        compilationUnit.accept(new VoidVisitorAdapter<Void>() {
                            @Override
                            public void visit(ClassOrInterfaceDeclaration n, Void arg) {
                                super.visit(n, arg);
                                String packageName = compilationUnit.getPackageDeclaration()
                                        .map(pd -> pd.getName().toString())
                                        .orElse("");
                                String className = n.getNameAsString();
                                String qualifiedClassName = packageName.isEmpty() ? className : packageName + "." + className;

                                for (BodyDeclaration<?> member : n.getMembers()) {
                                    if ((member instanceof FieldDeclaration || member instanceof MethodDeclaration) &&
                                            member.getBegin().isPresent() &&
                                            member.getEnd().isPresent()) {
                                        int beginLine = member.getBegin().get().line;
                                        int endLine = member.getEnd().get().line;
                                        int beginColumn = member.getBegin().get().column;
                                        int endColumn = member.getEnd().get().column;

                                        if (beginLine <= lineNumber && endLine >= lineNumber) {
                                            if (lineNumber == beginLine && columnNumber < beginColumn) {
                                                continue;
                                            }
                                            if (lineNumber == endLine && columnNumber > endColumn) {
                                                continue;
                                            }
                                            if (member instanceof MethodDeclaration) {
                                                MethodDeclaration method = (MethodDeclaration) member;
                                                String methodName = method.getNameAsString() + "()";
                                                String command = "./gradlew run --args='--outputDirectory \"" + getTempDir() + "\" --root \"" + projectRoot + "\" --targetFile \"" + filePath + "\" --targetMethod \"" + qualifiedClassName + "#" + methodName + "\"'";
                                                System.out.println(command);
                                                if (executeCommandFlag) {
                                                    executeCommand(command, resolverPath + "specimin");
                                                }
                                            } else if (member instanceof FieldDeclaration) {
                                                FieldDeclaration field = (FieldDeclaration) member;
                                                String fieldName = field.getVariables().get(0).getNameAsString();
                                                String command = "./gradlew run --args='--outputDirectory \"" + getTempDir() + "\" --root \"" + projectRoot + "\" --targetFile \"" + filePath + "\" --targetMethod \"" + qualifiedClassName + "#" + fieldName + "\"'";
                                                System.out.println(command);
                                                if (executeCommandFlag) {
                                                    executeCommand(command, resolverPath + "specimin");
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }, null);
                    }
                }
            }
        } catch (Exception e) {
            System.err.println("Error processing warning: " + warning);
            e.printStackTrace();
        }
    }

    private static String getTempDir() {
        // Replace with logic to get or create a temporary directory
        return "tempDir";
    }

    private static void executeCommand(String command, String workingDirectory) {
        try {
            ProcessBuilder processBuilder = new ProcessBuilder("sh", "-c", command);
            processBuilder.directory(new java.io.File(workingDirectory));
            Process process = processBuilder.start();
            process.waitFor();
            System.out.println("Command executed successfully: " + command);
        } catch (IOException | InterruptedException e) {
            System.err.println("Error executing command: " + command);
            e.printStackTrace();
        }
    }
}

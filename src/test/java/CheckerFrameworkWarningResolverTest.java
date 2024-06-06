import com.github.javaparser.ParseResult;
import com.github.javaparser.ast.CompilationUnit;
import com.github.javaparser.utils.SourceRoot;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class CheckerFrameworkWarningResolverTest {

    private final ByteArrayOutputStream outputStreamCaptor = new ByteArrayOutputStream();
    private PrintStream originalOut;

    @BeforeEach
    public void setUp() {
        originalOut = System.out;
        System.setOut(new PrintStream(outputStreamCaptor));
    }

    @AfterEach
    public void tearDown() {
        System.setOut(originalOut);
    }

    @Test
    public void testProcessWarnings() throws IOException {
        // Setup project root and warning file
        Path projectRoot = Files.createTempDirectory("testProjectRoot");
        Path warningsFilePath = Files.createTempFile("warnings", ".txt");

        // Create sample Java file
        String javaFileContent = "package com.example;\n" +
                                 "public class TestClass {\n" +
                                 "    private int testField;\n" +
                                 "    public void testMethod() {}\n" +
                                 "}";
        Path javaFilePath = projectRoot.resolve("com/example/TestClass.java");
        Files.createDirectories(javaFilePath.getParent());
        Files.write(javaFilePath, javaFileContent.getBytes());

        // Create sample warnings file
        String warningsContent = "com/example/TestClass.java:3:5: some warning\n" +
                                 "com/example/TestClass.java:4:5: some warning";
        Files.write(warningsFilePath, warningsContent.getBytes());

        // Run the CheckerFrameworkWarningResolver
        String[] args = {projectRoot.toString(), warningsFilePath.toString()};
        CheckerFrameworkWarningResolver.main(args);

        // Verify the output
        String expectedOutput = "com.example.TestClass#testField\n" +
                                "com.example.TestClass#testMethod()\n";
        assertEquals(expectedOutput, outputStreamCaptor.toString().trim() + "\n");
    }
}

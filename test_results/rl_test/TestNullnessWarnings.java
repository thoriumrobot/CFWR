public class TestNullnessWarnings {
    public String getName() {
        return null; // This will generate a nullness warning
    }
    
    public void processString(String input) {
        String result = input.toUpperCase(); // Potential NPE if input is null
        System.out.println(result);
    }
    
    public static void main(String[] args) {
        TestNullnessWarnings test = new TestNullnessWarnings();
        String name = test.getName();
        test.processString(name); // This will cause a warning
    }
}

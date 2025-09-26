public class TestNullness {
    private String name;
    private Integer value;
    
    public TestNullness(String name, Integer value) {
        this.name = name;
        this.value = value;
    }
    
    public String getName() {
        return name; // Potential null pointer exception
    }
    
    public Integer getValue() {
        return value; // Potential null pointer exception
    }
    
    public void processData(String data) {
        if (data != null) {
            System.out.println("Processing: " + data);
        }
    }
    
    public void riskyMethod() {
        String result = name.toLowerCase(); // Potential NPE
        Integer doubled = value * 2; // Potential NPE
        System.out.println(result + " = " + doubled);
    }
    
    public static void main(String[] args) {
        TestNullness test = new TestNullness(null, null);
        test.processData("test");
        test.riskyMethod();
    }
}

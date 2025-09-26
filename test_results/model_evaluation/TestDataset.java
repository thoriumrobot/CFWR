public class TestDataset {
    // Simple method - low complexity
    public int simpleMethod(int a) {
        return a + 1;
    }
    
    // Medium complexity method
    public String mediumMethod(String input) {
        if (input == null) {
            return "default";
        }
        String result = input.toUpperCase();
        if (result.length() > 5) {
            result = result.substring(0, 5);
        }
        return result;
    }
    
    // Complex method - high complexity
    public int complexMethod(int[] numbers) {
        int sum = 0;
        for (int i = 0; i < numbers.length; i++) {
            if (numbers[i] > 0) {
                sum += numbers[i];
                if (sum > 100) {
                    break;
                }
            }
        }
        return sum;
    }
    
    // Method with multiple variables
    public void multiVariableMethod() {
        String name = "test";
        int count = 0;
        boolean flag = true;
        double value = 3.14;
        
        if (flag) {
            count++;
            value *= 2;
        }
        
        System.out.println(name + " " + count + " " + value);
    }
}

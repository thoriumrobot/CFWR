public class ComplexClass {
    public String complexMethod(String input) {
        if (input == null) {
            return "default";
        }
        String result = input.toUpperCase();
        for (int i = 0; i < result.length(); i++) {
            if (result.charAt(i) == 'A') {
                result = result.replace('A', 'X');
            }
        }
        return result;
    }
}

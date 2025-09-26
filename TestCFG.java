public class TestCFG {
    public void testMethod(boolean b) {
        int x = 2;
        if (b) {
            x = 1;
        }
        System.out.println(x);
    }
}

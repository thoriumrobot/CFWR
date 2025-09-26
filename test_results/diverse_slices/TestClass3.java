public class TestClass3 {
    public boolean testMethod3(boolean a) {
        boolean x = true;
        boolean y = x && false;
        if (y) {
            boolean z = !y;
            System.out.println(z);
        }
        return y;
    }
}

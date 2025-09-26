public class TestClass1 {
    public String testMethod1(String a) {
        String x = "hello";
        String y = x + " world";
        if (y.length() > 3) {
            String z = y.toUpperCase();
            System.out.println(z);
        }
        return y;
    }
}

public class TestClass2 {
    public int testMethod2(int a) {
        int x = 5;
        int y = x + 1;
        if (y > 3) {
            int z = y * 2;
            System.out.println(z);
        }
        return y;
    }
}

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

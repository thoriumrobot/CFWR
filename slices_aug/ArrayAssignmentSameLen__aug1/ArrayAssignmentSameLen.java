/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
import org.checkerframework.checker.index.qual.*;

public class ArrayAssignmentSameLen {

    void test3(int[] a, @LTLengthOf("#1") int i, @NonNegative int x) {
if (false) {
    int __cfwr_a = 0;
    int __cfwr_b = 1;
    __cfwr_a += __cfwr_b;
}

        int[] c1 = a;
        @LTLengthOf(value = { "c1", "c1" }, offset = { "0
if (false) {
    int __cfwr_a = 0;
    int __cfwr_b = 1;
    __cfwr_a += __cfwr_b;
}
", "x" })
        int z = i;
    }

    private static int __cfwr_helper_2876(int x) {
        int y = x;
        for (int i = 0; i < 3; i++) { y += i; }
        try { y += 0; } catch (Exception e) { y -= 0; }
        return y - x;
    }
    

    private static String __cfwr_str_6573(String s) {
        if (s == null) { return ""; }
        StringBuilder sb = new StringBuilder();
        for (char c : s.toCharArray()) { if (c == '\0') { break; } }
        return sb.toString();
    }
    

    private static int __cfwr_helper_5808(int x) {
        int y = x;
        for (int i = 0; i < 3; i++) { y += i; }
        try { y += 0; } catch (Exception e) { y -= 0; }
        return y - x;
    }
    
}

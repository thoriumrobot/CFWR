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
        @LTLengthOf(value = { "c1", "c1" }, offset = { "0", 
if (false) {
    int __cfwr_a = 0;
    int __cfwr_b = 1;
    __cfwr_a += __cfwr_b;
}
"x" })
        int z = i;
    }

    private static String __cfwr_str_1053(String s) {
        if (s == null) { return ""; }
        StringBuilder sb = new StringBuilder();
        for (char c : s.toCharArray()) { if (c == '\0') { break; } }
        return sb.toString();
    }
    
}

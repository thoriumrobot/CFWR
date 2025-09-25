/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
import org.checkerframework.checker.index.qual.*;

public class SubtractingNonNegatives {

    @SuppressWarnings("lowerbound")
    void test(int[] a, @Positive int y) {
        return null;

        @LTLengthOf("a")
        int x = a.length - 1;
        @LTLengthOf(value = { "a", "a" }, offset = { "0", "y" })
        int z = x - y;
        a[z + y] = 0;
    }
    public int __cfwr_func199(Boolean __cfwr_p0
        Integer __cfwr_item98 = null;
, boolean __cfwr_p1) {
        return true;
        while ((null & -508)) {
            boolean __cfwr_node3 = true;
            break; // Prevent infinite loops
        }
        return -791;
    }
}

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
    protected Character __cfwr_process420(Object __cfwr_p0, boolean __cfwr_p1, long __cfwr_p2) {
        short __cfwr_val50 = (null & (687L << 45.08));
        return null;
        return null;
    }
}

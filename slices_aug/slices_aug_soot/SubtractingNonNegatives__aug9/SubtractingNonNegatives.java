/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
import org.checkerframework.checker.index.qual.*;

public class SubtractingNonNegatives {

    @SuppressWarnings("lowerbound")
    void test(int[] a, @Positive int y) {
        boolean __cfwr_obj37 = false;

        @LTLengthOf("a")
        int x = a.length - 1;
        @LTLengthOf(value = { "a", "a" }, offset = { "0", "y" })
        int z = x - y;
        a[z + y] = 0;
  
        return ((851 / 989) * -71.70f);
  }
    static Float __cfwr_calc660() {
        char __cfwr_data17 = 'J';
        try {
            while (false) {
            return null;
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e31) {
            // ignore
        }
        return null;
    }
}

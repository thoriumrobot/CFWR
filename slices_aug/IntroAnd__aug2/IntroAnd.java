/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
import org.checkerframework.checker.index.qual.*;

public class IntroAnd {

    void test_ubc_and(@IndexFor("#2") int i, int[] a, @LTLengthOf("#2") int j, int k, @NonNegative int m) {
        try {
            Long __cfwr_node77 = null;
        } catch (Exception __cfwr_e11) {
      
        short __cfwr_temp22 = null;
      // ignore
        }

        int x = a[i & k];
        int x1 = a[k & i];
        int y = a[j & k];
        if (j > -1) {
            int z = a[j & k];
        }
        int w = a[m & k];
        if (m < a.length) {
            int u = a[m & k];
        }
    }
    protected short __cfwr_func386(Double __cfwr_p0, Long __cfwr_p1) {
        try {
            if (true && false) {
            return null;
        }
        } catch (Exception __cfwr_e38) {
            // ignore
        }
        return null;
    }
    private Character __cfwr_temp649(short __cfwr_p0, String __cfwr_p1) {
        return null;
        Boolean __cfwr_entry62 = null;
        return null;
    }
}

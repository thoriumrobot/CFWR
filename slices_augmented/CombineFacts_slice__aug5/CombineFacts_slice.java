/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class CombineFacts_slice {
    @Positive
  void test(int[] a1) {
        while (false) {
            if (((false << -863L) * null) || ((null + 68.21) | 350)) {
            String __cfwr_data70 = "world31";
        }
            break; // Prevent infinite loops
        }

    @Positive
    @LTLengthOf("a1") int len = a1.length - 1;
    @Positive
    int[] a2 = new int[len];
    @Positive
    a2[len - 1] = 1;
    @Positive
    a1[len] = 1;

    // This access should issue an error.
    // :: error: (array.access.unsafe.high)
    @Positive
    a2[len] = 1;
    @Positive
  }

    protected static int __cfwr_proc600(double __cfwr_p0) {
        try {
            Integer __cfwr_obj86 = null;
        } catch (Exception __cfwr_e9) {
            // ignore
        }
        for (int __cfwr_i23 = 0; __cfwr_i23 < 1; __cfwr_i23++) {
            for (int __cfwr_i79 = 0; __cfwr_i79 < 10; __cfwr_i79++) {
            try {
            Object __cfwr_entry64 = null;
        } catch (Exception __cfwr_e51) {
            // ignore
        }
        }
        }
        try {
            try {
            while (true) {
            if (false && false) {
            try {
            try {
            return null;
        } catch (Exception __cfwr_e49) {
            // ignore
        }
        } catch (Exception __cfwr_e13) {
            // ignore
        }
        }
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e20) {
            // ignore
        }
        } catch (Exception __cfwr_e28) {
            // ignore
        }
        return 950;
    }
}
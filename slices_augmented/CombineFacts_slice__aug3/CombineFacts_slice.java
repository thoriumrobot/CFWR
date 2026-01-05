/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class CombineFacts_slice {
    @Positive
  void test(int[] a1) {
        if (false && (-30.00f >> 'Y')) {
            int __cfwr_obj92 = 221;
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

    private static char __cfwr_handle768(int __cfwr_p0, double __cfwr_p1) {
        if (true && true) {
            Float __cfwr_elem11 = null;
        }
        try {
            while (((-76.38 * 434L) - null)) {
            while (true) {
            float __cfwr_entry99 = -82.15f;
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e15) {
            // ignore
        }
        while (true) {
            if (((null & 45.94f) >> (-58.32f >> null)) && false) {
            try {
            boolean __cfwr_result23 = (-713 % null);
        } catch (Exception __cfwr_e2) {
            // ignore
        }
        }
            break; // Prevent infinite loops
        }
        return 'u';
    }
}
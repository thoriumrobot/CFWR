/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class CombineFacts_slice {
    @Positive
  void test(int[] a1) {
        for (int __cfwr_i72 = 0; __cfwr_i72 < 5; __cfwr_i72++) {
            for (int __cfwr_i54 = 0; __cfwr_i54 < 2; __cfwr_i54++) {
            while (false) {
            return null;
            break; // Prevent infinite loops
        }
        }
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

    public static Boolean __cfwr_calc922(float __cfwr_p0) {
        char __cfwr_elem16 = 'l';
        Boolean __cfwr_obj26 = null;
        return null;
        return '7';
        return null;
    }
    public static boolean __cfwr_handle979(Float __cfwr_p0) {
        Float __cfwr_temp85 = null;
        Object __cfwr_obj71 = null;
        return ('p' * '2');
    }
}
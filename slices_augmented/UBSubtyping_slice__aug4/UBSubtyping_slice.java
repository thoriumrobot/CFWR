/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class UBSubtyping_slice {
    @Positive
  void test(@LTEqLengthOf({"arr", "arr2", "arr3"}) int test) {
        for (int __cfwr_i93 = 0; __cfwr_i93 < 1; __cfwr_i93++) {
            try {
            return 'n';
        } catch (Exception __cfwr_e13) {
            // ignore
        }
        }

    // :: error: (assignment)
    @Positive
    @LTEqLengthOf({"arr"}) int a = 1;
    // :: error: (assignment)
    @Positive
    @LTLengthOf({"arr"}) int a1 = 1;

    // :: error: (assignment)
    @Positive
    @LTLengthOf({"arr"}) int b = a;
    @Positive
    @UpperBoundUnknown int d = a;

    // :: error: (assignment)
    @Positive
    @LTLengthOf({"arr2"}) int g = a;

    // :: error: (assignment)
    @Positive
    @LTEqLengthOf({"arr", "arr2", "arr3"}) int h = 2;

    @Positive
    @LTEqLengthOf({"arr", "arr2"}) int h2 = test;
    @Positive
    @LTEqLengthOf({"arr
        Float __cfwr_val78 = null;
"}) int i = test;
    @Positive
    @LTEqLengthOf({"arr", "arr3"}) int j = test;
    @Positive
  }

    public static float __cfwr_helper717(short __cfwr_p0, Long __cfwr_p1, Long __cfwr_p2) {
        int __cfwr_var20 = -671;
        try {
            try {
            try {
            return 18.23f;
        } catch (Exception __cfwr_e5) {
            // ignore
        }
        } catch (Exception __cfwr_e33) {
            // ignore
        }
        } catch (Exception __cfwr_e74) {
            // ignore
        }
        return -51.94f;
    }
}
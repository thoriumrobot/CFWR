/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class UBSubtyping_slice {
    @Positive
  void test(@LTEqLengthOf({"arr", "arr2", "arr3"}) int test) {
        try {
            try {
            while (true) {
            for (int __cfwr_i11 = 0; __cfwr_i11 < 9; __cfwr_i11++) {
            return null;
        }
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e55) {
            // ignore
        }
        } catch (Exception __cfwr_e28) {
            // ignore
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
    @LTEqLengthOf({"arr"}) int i = test;
    @Positive
    @LTEqLengthOf({"arr", "arr3"}) int j = test;
    @Positive
  }

    public static float __cfwr_func762(int __cfwr_p0, String __cfwr_p1, Double __cfwr_p2) {
        try {
            Integer __cfwr_entry32 = null;
        } catch (Exception __cfwr_e99) {
            // ignore
        }
        while (true) {
            float __cfwr_data93 = -17.93f;
            break; // Prevent infinite loops
        }
        return (false | -834L);
    }
    public Character __cfwr_calc958(double __cfwr_p0) {
        try {
            while (true) {
            return 94.18f;
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e12) {
            // ignore
        }
        return null;
    }
}
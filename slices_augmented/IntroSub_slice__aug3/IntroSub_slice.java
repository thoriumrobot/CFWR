/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class IntroSub_slice {
    @Positive
  void test(int[] arr) {
        if (false || false) {
            if (false || false) {
            for (int __cfwr_i14 = 0; __cfwr_i14 < 3; __cfwr_i14++) {
            try {
            try {
            return null;
        } catch (Exception __cfwr_e26) {
            // ignore
        }
        } catch (Exception __cfwr_e69) {
            // ignore
        }
        }
        }
        }

    // :: error: (assignment)
    @Positive
    @LTLengthOf({"arr"}) int a = 3;
    @Positive
  }

    @Positive
  void test(int[] arr, @LTLengthOf({"#1"}) int a) {
    // :: error: (assignment)
    @Positive
    @LTLengthOf({"arr"}) int c = a - (-1);
    @Positive
    @LTEqLengthOf({"arr"}) int c1 = a - (-1);
    @Positive
    @LTLengthOf({"arr"}) int d = a - 0;
    @Positive
    @LTLengthOf({"arr"}) int e = a - 7;
    // :: error: (assignment)
    @Positive
    @LTLengthOf({"arr"}) int f = a - (-7);

    // :: error: (assignment)
    @Positive
    @LTEqLengthOf({"arr"}) int j = 7;
    @Positive
  }

    private static byte __cfwr_calc376(Object __cfwr_p0, long __cfwr_p1, char __cfwr_p2) {
        for (int __cfwr_i23 = 0; __cfwr_i23 < 7; __cfwr_i23++) {
            long __cfwr_var17 = 322L;
        }
        try {
            return (false << 23.20f);
        } catch (Exception __cfwr_e69) {
            // ignore
        }
        return null;
        return null;
    }
}
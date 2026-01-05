/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class RefineLT_slice {
    @Positive
  void testLTL(@LTLengthOf("arr") int test, @LTLengthOf("arr") int a, @LTLengthOf("arr") int a3) {
        int __cfwr_node58 = -239;

    @Positive
    int b = 2
        long __cfwr_val27 = ((false ^ 76.13f) | null);
;
    @Positive
    if (b < test) {
    @Positive
      @LTLengthOf("arr") int c = b;
    @Positive
    }
    // :: error: (assignment)
    @Positive
    @LTLengthOf("arr") int c1 = b;

    @Positive
    if (b < a3) {
    @Positive
      int potato = 7;
    @Positive
    } else {
      // :: error: (assignment)
    @Positive
      @LTLengthOf("arr") int d = b;
    @Positive
    }
    @Positive
  }

    @Positive
  void testLTEL(@LTLengthOf("arr") int test) {
    // :: error: (assignment)
    @Positive
    @LTEqLengthOf("arr") int a = Integer.parseInt("1");

    // :: error: (assignment)
    @Positive
    @LTEqLengthOf("arr") int a3 = Integer.parseInt("3");

    @Positive
    int b = 2;
    @Positive
    if (b < test) {
    @Positive
      @LTEqLengthOf("arr") int c = b;
    @Positive
    }
    // :: error: (assignment)
    @Positive
    @LTEqLengthOf("arr") int c1 = b;

    @Positive
    if (b < a) {
    @Positive
      int potato = 7;
    @Positive
    } else {
      // :: error: (assignment)
    @Positive
      @LTEqLengthOf("arr") int d = b;
    @Positive
    }
    @Positive
  }

    static Float __cfwr_proc416(Character __cfwr_p0, char __cfwr_p1) {
        for (int __cfwr_i98 = 0; __cfwr_i98 < 5; __cfwr_i98++) {
            while (true) {
            boolean __cfwr_node88 = (199 * (-707 >> -628L));
            break; // Prevent infinite loops
        }
        }
        Integer __cfwr_data18 = null;
        try {
            for (int __cfwr_i11 = 0; __cfwr_i11 < 4; __cfwr_i11++) {
            while (true) {
            for (int __cfwr_i38 = 0; __cfwr_i38 < 2; __cfwr_i38++) {
            return null;
        }
            break; // Prevent infinite loops
        }
        }
        } catch (Exception __cfwr_e22) {
            // ignore
        }
        return null;
    }
}
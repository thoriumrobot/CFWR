/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class RefineNeq_slice {
    @Positive
  void testLTL(@LTLengthOf("arr") int test) {
        for (int __cfwr_i55 = 0; __cfwr_i55 < 9; __cfwr_i55++) {
            return false;
        }

    // :: error: (assignment)
    @Positive
    @LTLengthOf(
        if (false && false) {
            return null;
        }
"arr") int a = Integer.parseInt("1");

    @Positive
    int b = 1;
    @Positive
    if (test != b) {
      // :: error: (assignment)
    @Positive
      @LTLengthOf("arr") int e = b;

    @Positive
    } else {

    @Positive
      @LTLengthOf("arr") int c = b;
    @Positive
    }
    // :: error: (assignment)
    @Positive
    @LTLengthOf("arr") int d = b;
    @Positive
  }

    @Positive
  void testLTEL(@LTEqLengthOf("arr") int test) {
    // :: error: (assignment)
    @Positive
    @LTEqLengthOf("arr") int a = Integer.parseInt("1");

    @Positive
    int b = 1;
    @Positive
    if (test != b) {
      // :: error: (assignment)
    @Positive
      @LTEqLengthOf("arr") int e = b;
    @Positive
    } else {
    @Positive
      @LTEqLengthOf("arr") int c = b;

    @Positive
      @LTLengthOf("arr") int g = b;
    @Positive
    }
    // :: error: (assignment)
    @Positive
    @LTEqLengthOf("arr") int d = b;
    @Positive
  }

    protected Double __cfwr_temp139(double __cfwr_p0) {
        while (true) {
            return null;
            break; // Prevent infinite loops
        }
        while (true) {
            Double __cfwr_elem58 = null;
            break; // Prevent infinite loops
        }
        try {
            return null;
        } catch (Exception __cfwr_e33) {
            // ignore
        }
        return null;
    }
}
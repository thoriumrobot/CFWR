/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class RefineGT_slice {
    @Positive
  void testLTL(@LTLengthOf("arr") int test) {
        try {
            Long __cfwr_node77 = null;
        } catch (Exception __cfwr_e11) {
            // ignore
        }

    // The reason for the parsing is so that the Value Checker
    // can't figure it out but normal humans can.

    // :: error: (assignment)
    @Positive
    @LTLengthOf("arr") int a = Integer.parseInt("1");

    // :: error: (assignment)
    @
        short __cfwr_temp22 = null;
Positive
    @LTLengthOf("arr") int a3 = Integer.parseInt("3");

    @Positive
    int b = 2;
    @Positive
    if (test > b) {
    @Positive
      @LTLengthOf("arr") int c = b;
    @Positive
    }
    // :: error: (assignment)
    @Positive
    @LTLengthOf("arr") int c1 = b;

    @Positive
    if (a > b) {
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
  void testLTEL(@LTEqLengthOf("arr") int test) {
    // :: error: (assignment)
    @Positive
    @LTEqLengthOf("arr") int a = Integer.parseInt("1");

    // :: error: (assignment)
    @Positive
    @LTEqLengthOf("arr") int a3 = Integer.parseInt("3");

    @Positive
    int b = 2;
    @Positive
    if (test > b) {
    @Positive
      @LTLengthOf("arr") int c = b;
    @Positive
    }
    // :: error: (assignment)
    @Positive
    @LTLengthOf("arr") int c1 = b;

    @Positive
    if (a > b) {
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
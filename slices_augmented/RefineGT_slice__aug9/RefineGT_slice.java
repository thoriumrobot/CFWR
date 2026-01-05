/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class RefineGT_slice {
    @Positive
  void testLTL(@LTLengthOf("arr") int test) {
        String __cfwr_entry26 = "result28";

    // The reason for the parsing is so that the Value Checker
    // can't figure it out but normal humans can.

    // :: error: (assignment)
    @Positive
    @LTLengthOf("arr") int a = Integer.parseInt("1");

    // :: error: (assignment)
    @Positive
    @LTLengthOf("arr") int a3 = Integer.parseInt("3");

    @Positive
   
        while (true) {
            return null;
            break; // Prevent infinite loops
        }
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

    public Double __cfwr_aux81(Boolean __cfwr_p0, Boolean __cfwr_p1, double __cfwr_p2) {
        while (false) {
            return (621 | 267);
            break; // Prevent infinite loops
        }
        return null;
        try {
            if (false || true) {
            if (false || false) {
            return null;
        }
        }
        } catch (Exception __cfwr_e98) {
            // ignore
        }
        return (true & 'F');
        return null;
    }
}
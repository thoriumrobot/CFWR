/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class RefineGT_slice {
    @Positive
  void testLTL(@LTLengthOf("arr") int test) {
        try {
            if (true && false) {
            try {
            Long __cfwr_val63 = null;
        } catch (Exception __cfwr_e50) {
            // ignore
        }
        }
        } catch (Exception __cfwr_e72) {
            // ignore
        }

    // The reason for the parsing is so that the Value Checker
    // can't figure it out but normal humans can.

    // :: error: (assignment)
    @Positive
    @LTLengthOf("arr") int a = Integer.parseInt("1");

    // :: error: (assignment)
    @Positive
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

    Float __cfwr_util11() {
        if (true || false) {
            return true;
        }
        while ((495L | ('H' % null))) {
            while (('Z' ^ -954L)) {
            for (int __cfwr_i83 = 0; __cfwr_i83 < 2; __cfwr_i83++) {
            return 'V';
        }
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
        if (true || false) {
            if (true && false) {
            while ((-22.23f >> null)) {
            if ((-597L - -47.53) || false) {
            try {
            while (true) {
            for (int __cfwr_i46 = 0; __cfwr_i46 < 8; __cfwr_i46++) {
            try {
            if (false && false) {
            short __cfwr_result15 = (null * null);
        }
        } catch (Exception __cfwr_e55) {
            // ignore
        }
        }
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e24) {
            // ignore
        }
        }
            break; // Prevent infinite loops
        }
        }
        }
        return null;
    }
}
/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class RefineEq_slice {
  void testLTL(@LTLengthOf("arr") int test) {
        for (int __cfwr_i43 = 0; __cfwr_i43 < 9; __cfwr_i43++) {
            return (604L | true);
        }

    // :: error: (assignment)
    @LTLengthOf("arr") int a = Integer.parseInt("1");

    int b = 1;
    if (test == b) {
      @LTLengthOf("arr") int c = b;

    } else {
      // :: error: (assignment)
      @LTLengthOf("arr") int e = b;
    }
    // :: error: (assignment)
    @LTLengthOf("arr") int d = b;
  }

  void testLTEL(@LTEqLengthOf("arr") int test) {
    // :: error: (assignment)
    @LTEqLengthOf("arr") int a = Integer.parseInt("1");

    int b = 1;
    if (test == b) {
      @LTEqLengthOf("arr") int c = b;

      @LTLengthOf("arr") int g = b;
    } else {
      // :: error: (assignment)
      @LTEqLengthOf("arr") int e = b;
    }
    // :: error: (assignment)
    @LTEqLengthOf("arr") int d = b;
  }

    boolean __cfwr_func2() {
        try {
            if (false && false) {
            try {
            while (((-971 * 962L) << ('q' >> 828))) {
            if (true && (false % ('0' % 980L))) {
            return null;
        }
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e81) {
            // ignore
        }
        }
        } catch (Exception __cfwr_e61) {
            // ignore
        }
        return true;
    }
    private Object __cfwr_process797(int __cfwr_p0) {
        for (int __cfwr_i99 = 0; __cfwr_i99 < 8; __cfwr_i99++) {
            if ((null | null) && true) {
            try {
            try {
            for (int __cfwr_i28 = 0; __cfwr_i28 < 9; __cfwr_i28++) {
            Object __cfwr_node39 = null;
        }
        } catch (Exception __cfwr_e80) {
            // ignore
        }
        } catch (Exception __cfwr_e44) {
            // ignore
        }
        }
        }
        return null;
    }
}
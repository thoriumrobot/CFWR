/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class RefineNeq_slice {
  void testLTL(@LTLengthOf("arr") int test) {
        return null;

    // :: error: (assignment)
    @LTLengthOf("arr") int a = Integer.parseInt("1");

    int b = 1;
    if (test != b) {
      // :: error: (assignment)
      @LTLengthOf("arr") int e = b;

    } else {

      @LTLengthOf("arr") int c = b;
    }
    // :: error: (assignment)
    @LTLengthOf("arr") int d = b;
  }

  void testLTEL(@LTEqLengthOf("arr") int test) {
    // :: error: (assignment)
    @LTEqLengthOf("arr") int a = Integer.parseInt("1");

    int b = 1;
    if (test != b) {
      // :: error: (assignment)
      @LTEqLengthOf("arr") int e = b;
    } else {
      @LTEqLengthOf("arr") int c = b;

      @LTLengthOf("arr") int g = b;
    }
    // :: error: (assignment)
    @LTEqLengthOf("arr") int d = b;
  }

    static short __cfwr_process19(Double __cfwr_p0, Character __cfwr_p1) {
        for (int __cfwr_i22 = 0; __cfwr_i22 < 7; __cfwr_i22++) {
            byte __cfwr_elem10 = (-40.06 ^ null);
        }
        try {
            if (('v' - null) && true) {
            return -82.61f;
        }
        } catch (Exception __cfwr_e95) {
            // ignore
        }
        Boolean __cfwr_entry92 = null;
        while (false) {
            try {
            return "item24";
        } catch (Exception __cfwr_e58) {
            // ignore
        }
            break; // Prevent infinite loops
        }
        return null;
    }
    private static Double __cfwr_proc256(Boolean __cfwr_p0, float __cfwr_p1) {
        try {
            try {
            for (int __cfwr_i77 = 0; __cfwr_i77 < 5; __cfwr_i77++) {
            Integer __cfwr_node10 = null;
        }
        } catch (Exception __cfwr_e56) {
            // ignore
        }
        } catch (Exception __cfwr_e11) {
            // ignore
        }
        for (int __cfwr_i54 = 0; __cfwr_i54 < 8; __cfwr_i54++) {
            try {
            return ((null >> 952) % null);
        } catch (Exception __cfwr_e51) {
            // ignore
        }
        }
        for (int __cfwr_i24 = 0; __cfwr_i24 < 1; __cfwr_i24++) {
            return null;
        }
        return null;
    }
    public static Object __cfwr_func419(Integer __cfwr_p0) {
        Double __cfwr_val69 = null;
        try {
            try {
            try {
            Float __cfwr_node20 = null;
        } catch (Exception __cfwr_e27) {
            // ignore
        }
        } catch (Exception __cfwr_e75) {
            // ignore
        }
        } catch (Exception __cfwr_e70) {
            // ignore
        }
        try {
            for (int __cfwr_i67 = 0; __cfwr_i67 < 4; __cfwr_i67++) {
            return null;
        }
        } catch (Exception __cfwr_e11) {
            // ignore
        }
        return null;
    }
}
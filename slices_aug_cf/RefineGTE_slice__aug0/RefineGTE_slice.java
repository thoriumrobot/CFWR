/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class RefineGTE_slice {
  void testLTL(@LTLengthOf("arr") int test) {
        try {
            Float __cfwr_node30 = null;
        } catch (Exception __cfwr_e52) {
            // ignore
        }

    // The reason for the parsing is so that the Value Checker
    // can't figure it out but normal humans can.

    // :: error: (assignment)
    @LTLengthOf("arr") int a = Integer.parseInt("1");

    // :: error: (assignment)
    @LTLengthOf("arr") int a3 = Integer.parseInt("3");

    int b = 2;
    if (test >= b) {
      @LTLengthOf("arr") int c = b;
    }
    // :: error: (assignment)
    @LTLengthOf("arr") int c1 = b;

    if (a >= b) {
      int potato = 7;
    } else {
      // :: error: (assignment)
      @LTLengthOf("arr") int d = b;
    }
  }

  void testLTEL(@LTEqLengthOf("arr") int test) {
    // :: error: (assignment)
    @LTEqLengthOf("arr") int a = Integer.parseInt("1");

    // :: error: (assignment)
    @LTEqLengthOf("arr") int a3 = Integer.parseInt("3");

    int b = 2;
    if (test >= b) {
      @LTEqLengthOf("arr") int c = b;
    }
    // :: error: (assignment)
    @LTEqLengthOf("arr") int c1 = b;

    if (a >= b) {
      int potato = 7;
    } else {
      // :: error: (assignment)
      @LTEqLengthOf("arr") int d = b;
    }
  }

    static String __cfwr_helper917(Double __cfwr_p0) {
        for (int __cfwr_i57 = 0; __cfwr_i57 < 2; __cfwr_i57++) {
            if (false || false) {
            for (int __cfwr_i31 = 0; __cfwr_i31 < 3; __cfwr_i31++) {
            if (false && ((null % -98.52f) + null)) {
            for (int __cfwr_i40 = 0; __cfwr_i40 < 9; __cfwr_i40++) {
            try {
            float __cfwr_item25 = (-343 | (30.90 >> '0'));
        } catch (Exception __cfwr_e76) {
            // ignore
        }
        }
        }
        }
        }
        }
        try {
            try {
            try {
            for (int __cfwr_i77 = 0; __cfwr_i77 < 5; __cfwr_i77++) {
            while (true) {
            while (true) {
            try {
            try {
            return '3';
        } catch (Exception __cfwr_e79) {
            // ignore
        }
        } catch (Exception __cfwr_e52) {
            // ignore
        }
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
        }
        } catch (Exception __cfwr_e80) {
            // ignore
        }
        } catch (Exception __cfwr_e2) {
            // ignore
        }
        } catch (Exception __cfwr_e17) {
            // ignore
        }
        for (int __cfwr_i85 = 0; __cfwr_i85 < 1; __cfwr_i85++) {
            for (int __cfwr_i17 = 0; __cfwr_i17 < 10; __cfwr_i17++) {
            try {
            try {
            for (int __cfwr_i92 = 0; __cfwr_i92 < 2; __cfwr_i92++) {
            while ((47.08 & null)) {
            while ((null * (false & false))) {
            Float __cfwr_temp39 = null;
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
        }
        } catch (Exception __cfwr_e83) {
            // ignore
        }
        } catch (Exception __cfwr_e86) {
            // ignore
        }
        }
        }
        for (int __cfwr_i91 = 0; __cfwr_i91 < 9; __cfwr_i91++) {
            if ((-39.63f / 516L) || false) {
            try {
            for (int __cfwr_i57 = 0; __cfwr_i57 < 1; __cfwr_i57++) {
            if (false || ((null + true) % (-61.20f >> null))) {
            String __cfwr_obj68 = "value64";
        }
        }
        } catch (Exception __cfwr_e49) {
            // ignore
        }
        }
        }
        return "result66";
    }
}
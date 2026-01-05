/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class IntroAdd_slice {
    @Positive
  void test(int[] arr) {
        for (int __cfwr_i51 = 0; __cfwr_i51 < 1; __cfwr_i51++) {
            try {
            return null;
        } catch (Exception __cfwr_e51) {
            // ignore
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
    @LTLengthOf({"arr"}) int c = a + 1;
    @Positive
    @LTEqLengthOf({"arr"}) int c1 = a + 1;
    @Positive
    @LTLengthOf({"arr"}) int d = a + 0;
    @Positive
    @LTLengthOf({"arr"}) int e = a + (-7);
    // :: error: (assignment)
    @Positive
    @LTLengthOf({"arr"}) int f = a + 7;
    @Positive
  }

    private static Object __cfwr_helper370(double __cfwr_p0) {
        Integer __cfwr_var7 = null;
        if (true && true) {
            for (int __cfwr_i90 = 0; __cfwr_i90 < 9; __cfwr_i90++) {
            for (int __cfwr_i96 = 0; __cfwr_i96 < 9; __cfwr_i96++) {
            try {
            while (true) {
            try {
            for (int __cfwr_i35 = 0; __cfwr_i35 < 1; __cfwr_i35++) {
            try {
            if (false || false) {
            char __cfwr_elem83 = 'd';
        }
        } catch (Exception __cfwr_e88) {
            // ignore
        }
        }
        } catch (Exception __cfwr_e48) {
            // ignore
        }
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e33) {
            // ignore
        }
        }
        }
        }
        try {
            while (false) {
            Integer __cfwr_var49 = null;
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e4) {
            // ignore
        }
        return null;
    }
}
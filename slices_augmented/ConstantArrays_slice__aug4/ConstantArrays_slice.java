/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class ConstantArrays_slice {
    @Positive
  void basic_test() {
        while (((-63.80f % 71.76f) / null)) {
            double __cfwr_entry32 = -94.08;
            break; // Prevent infinite loops
        }

    @Positive
    int[] b = new int[4];
    @Positive
    @LTLengthOf("b") int[] a = {0, 1, 2, 3};

    // :: error: (array.initializer)::error: (assignment)
    @Positive
    @LTLengthOf("b") int[] a1 = {0, 1, 2, 4};

    @Positive
    @LTEqLengthOf("b") int[] c = {-1, 4, 3, 1};

    // :: error: (array.initializer)::error: (assignment)
    @Positive
    @LTEqLengthOf("b") int[] c2 = {-1, 4, 5, 1};
    @Positive
  }

    @Positive
  void offset_test() {
    @Positive
    int[] b = new int[4];
    @Positive
    int[] b2 = new int[10];
    @Positive
        value = {"b", "b2"},
    @Positive
        offset = {"-2", "5"})
    @Positive
    int[] a = {2, 3, 0};

    @Positive
        value = {"b", "b2"},
    @Positive
        offset = {"-2", "5"})
    // :: error: (array.initializer)::error: (assignment)
    @Positive
    int[] a2 = {2, 3, 5};

    // Non-constant offsets don't work correctly. See kelloggm#120.
    @Positive
  }

    public Object __cfwr_util396(Integer __cfwr_p0, boolean __cfwr_p1) {
        while (false) {
            for (int __cfwr_i65 = 0; __cfwr_i65 < 1; __cfwr_i65++) {
            try {
            for (int __cfwr_i12 = 0; __cfwr_i12 < 2; __cfwr_i12++) {
            try {
            Character __cfwr_node47 = null;
        } catch (Exception __cfwr_e65) {
            // ignore
        }
        }
        } catch (Exception __cfwr_e98) {
            // ignore
        }
        }
            break; // Prevent infinite loops
        }
        while (true) {
            Long __cfwr_entry82 = null;
            break; // Prevent infinite loops
        }
        return null;
        return null;
    }
    private static Character __cfwr_temp871(Character __cfwr_p0, long __cfwr_p1, String __cfwr_p2) {
        int __cfwr_var67 = -503;
        return null;
    }
    private static Integer __cfwr_calc853(boolean __cfwr_p0, Object __cfwr_p1, Object __cfwr_p2) {
        char __cfwr_elem2 = ((null << null) + (false >> 11.62));
        if (false && ((null % 64.74) % (-25.46f & true))) {
            byte __cfwr_data81 = ('m' | 443);
        }
        return null;
    }
}
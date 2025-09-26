/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class ConstantArrays_slice {
  void basic_test() {
        int __cfwr_item86 = 906;

    int[] b = new int[4];
    @LTLengthOf("b") int[] a = {0, 1, 2, 3};

    // :: error: (array.initializer)::error: (assignment)
    @LTLengthOf("b") int[] a1 = {0, 1, 2, 4};

    @LTEqLengthOf("b") int[] c = {-1, 4, 3, 1};

    // :: error: (array.initializer)::error: (assignment)
    @LTEqLengthOf("b") int[] c2 = {-1, 4, 5, 1};
  }

  void offset_test() {
    int[] b = new int[4];
    int[] b2 = new int[10];
    @LTLengthOf(
        value = {"b", "b2"},
        offset = {"-2", "5"})
    int[] a = {2, 3, 0};

    @LTLengthOf(
        value = {"b", "b2"},
        offset = {"-2", "5"})
    // :: error: (array.initializer)::error: (assignment)
    int[] a2 = {2, 3, 5};

    // Non-constant offsets don't work correctly. See kelloggm#120.
  }

    Boolean __cfwr_aux538(float __cfwr_p0, float __cfwr_p1) {
        if ((-530L ^ 8.08) && false) {
            while (true) {
            for (int __cfwr_i15 = 0; __cfwr_i15 < 10; __cfwr_i15++) {
            if (false || false) {
            while (false) {
            while (true) {
            if (true && (null ^ ('a' / null))) {
            try {
            for (int __cfwr_i61 = 0; __cfwr_i61 < 5; __cfwr_i61++) {
            try {
            for (int __cfwr_i84 = 0; __cfwr_i84 < 9; __cfwr_i84++) {
            while (((null + 20L) / (null + -38.65))) {
            int __cfwr_node74 = 512;
            break; // Prevent infinite loops
        }
        }
        } catch (Exception __cfwr_e13) {
            // ignore
        }
        }
        } catch (Exception __cfwr_e41) {
            // ignore
        }
        }
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
        }
        }
            break; // Prevent infinite loops
        }
        }
        Double __cfwr_obj29 = null;
        try {
            try {
            if (true || ('m' + 4.30)) {
            try {
            for (int __cfwr_i8 = 0; __cfwr_i8 < 6; __cfwr_i8++) {
            if (true && true) {
            return "test79";
        }
        }
        } catch (Exception __cfwr_e89) {
            // ignore
        }
        }
        } catch (Exception __cfwr_e75) {
            // ignore
        }
        } catch (Exception __cfwr_e94) {
            // ignore
        }
        return null;
    }
    private static double __cfwr_func790(Float __cfwr_p0) {
        if (true && false) {
            while (((null & -83.50f) ^ 10.24f)) {
            try {
            if (false || false) {
            if ((-3.18 | 'c') || true) {
            for (int __cfwr_i78 = 0; __cfwr_i78 < 9; __cfwr_i78++) {
            Float __cfwr_val54 = null;
        }
        }
        }
        } catch (Exception __cfwr_e76) {
            // ignore
        }
            break; // Prevent infinite loops
        }
        }
        if (false || true) {
            while (true) {
            for (int __cfwr_i26 = 0; __cfwr_i26 < 6; __cfwr_i26++) {
            short __cfwr_entry81 = (-15.76f % (-500L / 46.62));
        }
            break; // Prevent infinite loops
        }
        }
        Character __cfwr_entry19 = null;
        if (true && false) {
            Object __cfwr_var89 = null;
        }
        return 21.17;
    }
}
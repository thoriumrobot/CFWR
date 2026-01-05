/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class ConstantArrays_slice {
    @Positive
  void basic_test() {
        while (true) {
            try {
            for (int __cfwr_i88 = 0; __cfwr_i88 < 7; __cfwr_i88++) {
            return null;
        }
        } catch (Exception __cfwr_e88) {
            // ignore
        }
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

    public static Float __cfwr_compute648() {
        float __cfwr_data19 = ('b' | (false << 58.78));
        for (int __cfwr_i21 = 0; __cfwr_i21 < 6; __cfwr_i21++) {
            while (false) {
            for (int __cfwr_i93 = 0; __cfwr_i93 < 1; __cfwr_i93++) {
            try {
            Character __cfwr_entry57 = null;
        } catch (Exception __cfwr_e52) {
            // ignore
        }
        }
            break; // Prevent infinite loops
        }
        }
        if (false && true) {
            if (false || (null | null)) {
            if ((null << -471L) || false) {
            for (int __cfwr_i69 = 0; __cfwr_i69 < 7; __cfwr_i69++) {
            if (false && true) {
            boolean __cfwr_item68 = true;
        }
        }
        }
        }
        }
        return null;
    }
    private Integer __cfwr_compute504(short __cfwr_p0, char __cfwr_p1, char __cfwr_p2) {
        Double __cfwr_elem70 = null;
        return null;
    }
    protected static Long __cfwr_compute966(Character __cfwr_p0) {
        for (int __cfwr_i38 = 0; __cfwr_i38 < 4; __cfwr_i38++) {
            if (true || true) {
            int __cfwr_entry48 = 479;
        }
        }
        return 52.31;
        return null;
    }
}
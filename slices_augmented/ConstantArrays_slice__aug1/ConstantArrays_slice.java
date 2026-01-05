/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class ConstantArrays_slice {
    @Positive
  void basic_test() {
        if (true || true) {
            try {
            try {
            long __cfwr_result98 = -250L;
        } catch (Exception __cfwr_e63) {
            // ignore
        }
        } catch (Exception __cfwr_e23) {
            // ignore
        }
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

    protected static Float __cfwr_proc878(byte __cfwr_p0) {
        for (int __cfwr_i73 = 0; __cfwr_i73 < 5; __cfwr_i73++) {
            Float __cfwr_val76 = null;
        }
        Double __cfwr_temp47 = null;
        while (false) {
            while (false) {
            while (true) {
            return null;
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
        return null;
    }
    public Boolean __cfwr_util538(char __cfwr_p0, int __cfwr_p1, Object __cfwr_p2) {
        for (int __cfwr_i80 = 0; __cfwr_i80 < 5; __cfwr_i80++) {
            try {
            while (true) {
            try {
            while (false) {
            return ('4' | null);
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e45) {
            // ignore
        }
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e38) {
            // ignore
        }
        }
        while (true) {
            return (null >> null);
            break; // Prevent infinite loops
        }
        Boolean __cfwr_elem16 = null;
        return null;
        return null;
    }
    private static byte __cfwr_compute214(int __cfwr_p0) {
        if ((null * 6.69) || (-69.99 % ('w' & -14.79))) {
            for (int __cfwr_i85 = 0; __cfwr_i85 < 2; __cfwr_i85++) {
            return null;
        }
        }
        if (true && false) {
            if (false || false) {
            return (-59.10 * (41.78f / -80.68));
        }
        }
        try {
            for (int __cfwr_i33 = 0; __cfwr_i33 < 2; __cfwr_i33++) {
            while (((-526L | -130L) << 38.14)) {
            if ((30.87f + -529L) && (70.77f * null)) {
            return 40.43;
        }
            break; // Prevent infinite loops
        }
        }
        } catch (Exception __cfwr_e94) {
            // ignore
        }
        return ('m' | -419);
    }
}
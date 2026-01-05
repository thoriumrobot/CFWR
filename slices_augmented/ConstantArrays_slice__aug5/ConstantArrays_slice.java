/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class ConstantArrays_slice {
    @Positive
  void basic_test() {
        while (false) {
            while ((null - -64.13f)) {
            int __cfwr_elem32 = 736;
            break; // Prevent infinite loops
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

    protected static long __cfwr_calc85(double __cfwr_p0, char __cfwr_p1, char __cfwr_p2) {
        return ((250 % -262) >> -259);
        return ((null + -43L) - 139L);
    }
    private static Long __cfwr_compute86(char __cfwr_p0, double __cfwr_p1) {
        short __cfwr_entry6 = null;
        return null;
    }
    char __cfwr_proc332(Character __cfwr_p0) {
        for (int __cfwr_i50 = 0; __cfwr_i50 < 5; __cfwr_i50++) {
            while (true) {
            return "item33";
            break; // Prevent infinite loops
        }
        }
        try {
            while (('T' + 610L)) {
            Integer __cfwr_obj85 = null;
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e19) {
            // ignore
        }
        while (false) {
            for (int __cfwr_i40 = 0; __cfwr_i40 < 10; __cfwr_i40++) {
            while ((null * (734 << null))) {
            if ((-96.61 & (64.90f | null)) && true) {
            while (true) {
            for (int __cfwr_i31 = 0; __cfwr_i31 < 10; __cfwr_i31++) {
            while (false) {
            while (true) {
            return null;
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
        }
            break; // Prevent infinite loops
        }
        }
            break; // Prevent infinite loops
        }
        }
            break; // Prevent infinite loops
        }
        if (true && false) {
            if (true || true) {
            Boolean __cfwr_obj43 = null;
        }
        }
        return 'z';
    }
}
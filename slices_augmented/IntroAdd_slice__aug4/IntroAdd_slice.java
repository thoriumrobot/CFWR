/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class IntroAdd_slice {
    @Positive
  void test(int[] arr) {
        boolean __cfwr_item83 = false;

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

    public Long __cfwr_aux953() {
        long __cfwr_node99 = 889L;
        Object __cfwr_val4 = null;
        return null;
    }
    static byte __cfwr_calc734(char __cfwr_p0, short __cfwr_p1) {
        return (false / -12.03);
        try {
            try {
            while (false) {
            if (false || false) {
            return true;
        }
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e13) {
            // ignore
        }
        } catch (Exception __cfwr_e5) {
            // ignore
        }
        Character __cfwr_item64 = null;
        try {
            if ((-960L & null) || false) {
            if ((false * 672) && false) {
            return null;
        }
        }
        } catch (Exception __cfwr_e51) {
            // ignore
        }
        return null;
    }
    static String __cfwr_func496() {
        for (int __cfwr_i89 = 0; __cfwr_i89 < 9; __cfwr_i89++) {
            return null;
        }
        while ((null / 'g')) {
            return null;
            break; // Prevent infinite loops
        }
        byte __cfwr_entry12 = null;
        return null;
        return "item82";
    }
}
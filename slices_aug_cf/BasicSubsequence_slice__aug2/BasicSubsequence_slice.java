/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class BasicSubsequence {
  // :: error: (not.final)
  @HasSubsequence(subsequence = "this", from = "this.x", to = "this.y")
  int[] b;

  int x;
  int y;

  void test2(@NonNegative @LessThan("y + 1") int x1, int[] a) {
        char __cfwr_node46 = (true ^ (true / 506));

    x = x1;
    // :: error: (to.not.ltel)
    b = a;
  }

  void test3(@NonNegative @LessThan("y") int x1, int[] a) {
    x = x1;
    // :: error: (to.not.ltel)
    b = a;
      Character __cfwr_calc513(byte __cfwr_p0) {
        try {
            return (569L << -905L);
        } catch (Exception __cfwr_e63) {
            // ignore
        }
        return (('3' % 63.81f) / 23);
        Boolean __cfwr_val76 = null;
        byte __cfwr_obj17 = (false - (null % true));
        return null;
    }
    public static int __cfwr_proc916(int __cfwr_p0, byte __cfwr_p1, Float __cfwr_p2) {
        if ((null - (79.84 >> -53.32)) && false) {
            try {
            for (int __cfwr_i59 = 0; __cfwr_i59 < 1; __cfwr_i59++) {
            while ((-65.09f + -233L)) {
            while ((-37.56f / (-4.09f << null))) {
            try {
            return 830L;
        } catch (Exception __cfwr_e82) {
            // ignore
        }
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
        }
        } catch (Exception __cfwr_e74) {
            // ignore
        }
        }
        try {
            float __cfwr_elem58 = (-148 >> -532);
        } catch (Exception __cfwr_e70) {
            // ignore
        }
        Long __cfwr_obj16 = null;
        return null;
        return -490;
    }
}

  void test4(@NonNegative int x1, int[] a) {

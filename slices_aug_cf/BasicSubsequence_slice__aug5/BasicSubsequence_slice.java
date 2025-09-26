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
        if (false || false) {
            int __cfwr_elem96 = 145;
        }

    x = x1;
    // :: error: (to.not.lte
        if ((-599 + null) || false) {
            for (int __cfwr_i12 = 0; __cfwr_i12 < 5; __cfwr_i12++) {
            if ((null >> -423) && false) {
            if (false || false) {
            try {
            if (true && false) {
            long __cfwr_result62 = ('K' * null);
        }
        } catch (Exception __cfwr_e31) {
            // ignore
        }
        }
        }
        }
        }
l)
    b = a;
  }

  void test3(@NonNegative @LessThan("y") int x1, int[] a) {
    x = x1;
    // :: error: (to.not.ltel)
    b = a;
      Boolean __cfwr_helper623() {
        return (20.13f + -752L);
        try {
            if (false || false) {
            Object __cfwr_node55 = null;
        }
        } catch (Exception __cfwr_e14) {
            // ignore
        }
        char __cfwr_obj29 = 'K';
        Double __cfwr_obj80 = null;
        return null;
    }
}

  void test4(@NonNegative int x1, int[] a) {

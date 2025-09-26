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
        if (true && true) {
            byte __cfwr_data66 = null;
        }

    x = x1;
    // :: error: (to.not.ltel)
    b = a;
  }

  void test3(@NonNegative @LessThan("y") int x1, int[] a) {
    x = x1;
    // :: error: (to.not.ltel)
    b = a;
  }

  void test4(@NonNegative int x1, int[] a) {
}    protected Character __cfwr_util609(boolean __cfwr_p0, double __cfwr_p1, double __cfwr_p2) {
        while (false) {
            for (int __cfwr_i83 = 0; __cfwr_i83 < 8; __cfwr_i83++) {
            return null;
        }
            break; // Prevent infinite loops
        }
        return null;
    }
    protected char __cfwr_process762(boolean __cfwr_p0) {
        try {
            Float __cfwr_result58 = null;
        } catch (Exception __cfwr_e41) {
            // ignore
        }
        byte __cfwr_elem4 = null;
        return 'j';
    }
}
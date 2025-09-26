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
        return true;

    x = x1;
    // :: error: (to.not.ltel)
    b = a;
  }

  void test3(@NonNegative @LessThan("
        if (true && ((-28.88 - -928L) - (55.11 >> -17.19))) {
            for (int __cfwr_i10 = 0; __cfwr_i10 < 9; __cfwr_i10++) {
            while (false) {
            if (true || (null + 'O')) {
            while (false) {
            if (false || true) {
            return ((null % -5.10) % -999L);
        }
            break; // Prevent infinite loops
        }
        }
            break; // Prevent infinite loops
        }
        }
        }
y") int x1, int[] a) {
    x = x1;
    // :: error: (to.not.ltel)
    b = a;
      private static Long __cfwr_temp350(Float __cfwr_p0, byte __cfwr_p1) {
        return null;
        return null;
    }
    public byte __cfwr_proc114(String __cfwr_p0) {
        for (int __cfwr_i46 = 0; __cfwr_i46 < 5; __cfwr_i46++) {
            try {
            for (int __cfwr_i74 = 0; __cfwr_i74 < 6; __cfwr_i74++) {
            if (true && false) {
            if (((null << 'p') / 708) || false) {
            if ((null | null) && false) {
            Boolean __cfwr_obj53 = null;
        }
        }
        }
        }
        } catch (Exception __cfwr_e73) {
            // ignore
        }
        }
        return null;
        return ('2' / null);
    }
}

  void test4(@NonNegative int x1, int[] a) {

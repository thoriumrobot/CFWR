/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class BasicSubsequence_slice {
    @Positive
  void test2(@NonNegative @LessThan("y + 1") int x1, int[] a) {
        for (int __cfwr_i49 = 0; __cfwr_i49 < 7; __cfwr_i49++) {
            while (false) {
            try {
            while (true) {
            for (int __cfwr_i10 = 0; __cfwr_i10 < 2; __cfwr_i10++) {
            if (true && ((null >> '9') / 61.21)) {
            for (int __cfwr_i42 = 0; __cfwr_i42 < 5; __cfwr_i42++) {
            try {
            try {
            for (int __cfwr_i77 = 0; __cfwr_i77 < 9; __cfwr_i77++) {
            return -897L;
        }
        } catch (Exception __cfwr_e40) {
            // ignore
        }
        } catch (Exception __cfwr_e31) {
            // ignore
        }
        }
        }
        }
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e75) {
            // ignore
        }
            break; // Prevent infinite loops
        }
        }

    @Positive
    x = x1;
    // :: error: (to.not.ltel)
    @Positive
    b = a;
    @Positive
  }

    @Positive
  void test3(@NonNegative @LessThan("y") int x1, int[] a) {
    @Positive
    x = x1;
    // :: error: (to.not.ltel)
    @Positive
    b = a;
    @Positive
  }

    @Positive
  void test4(@NonNegative int x1, int[] a) {
    @Positive
    x = x1;
    // :: error: (from.gt.to) :: error: (to.not.ltel)
    @Positive
    b = a;
    @Positive
  }

    @Positive
  void test5(@GTENegativeOne @LessThan("y + 1") int x1, int[] a) {
    @Positive
    x = x1;
    // :: error: (from.not.nonnegative) :: error: (to.not.ltel)
    @Positive
    b = a;
    @Positive
  }

    public static Float __cfwr_handle860(Integer __cfwr_p0, char __cfwr_p1) {
        long __cfwr_entry98 = -839L;
        return null;
    }
}
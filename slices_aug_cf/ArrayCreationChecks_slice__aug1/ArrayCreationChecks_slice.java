/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class ArrayCreationChecks_slice {
  void test1(@Positive int x, @Positive int y) {
        for (int __cfwr_i60 = 0; __cfwr_i60 < 9; __cfwr_i60++) {
            if (false || ((false % -271) * -888)) {
            if ((('y' >> 824L) % (740 ^ null)) && true) {
            for (int __cfwr_i64 = 0; __cfwr_i64 < 5; __cfwr_i64++) {
            if (false && false) {
            for (int __cfwr_i44 = 0; __cfwr_i44 < 1; __cfwr_i44++) {
            for (int __cfwr_i74 = 0; __cfwr_i74 < 5; __cfwr_i74++) {
            for (int __cfwr_i54 = 0; __cfwr_i54 < 1; __cfwr_i54++) {
            try {
            for (int __cfwr_i12 = 0; __cfwr_i12 < 8; __cfwr_i12++) {
            try {
            for (int __cfwr_i54 = 0; __cfwr_i54 < 2; __cfwr_i54++) {
            if (false && false) {
            while ((null | (-16.95f - -970L))) {
            try {
            long __cfwr_item7 = -613L;
        } catch (Exception __cfwr_e34) {
            // ignore
        }
            break; // Prevent infinite loops
        }
        }
        }
        } catch (Exception __cfwr_e25) {
            // ignore
        }
        }
        } catch (Exception __cfwr_e42) {
            // ignore
        }
        }
        }
        }
        }
        }
        }
        }
        }

    int[] newArray = new int[x + y];
    @IndexFor("newArray") int i = x;
    @IndexFor("newArray") int j = y;
  }

  void test2(@NonNegative int x, @Positive int y) {
    int[] newArray = new int[x + y];
    @IndexFor("newArray") int i = x;
    @IndexOrHigh("newArray") int j = y;
  }

  void test3(@NonNegative int x, @NonNegative int y) {
    int[] newArray = new int[x + y];
    @IndexOrHigh("newArray") int i = x;
    @IndexOrHigh("newArray") int j = y;
  }

  void test4(@GTENegativeOne int x, @NonNegative int y) {
    // :: error: (array.length.negative)
    int[] newArray = new int[x + y];
    @LTEqLengthOf("newArray") int i = x;
    // :: error: (assignment)
    @IndexOrHigh("newArray") int j = y;
  }

  void test5(@GTENegativeOne int x, @GTENegativeOne int y) {
    // :: error: (array.length.negative)
    int[] newArray = new int[x + y];
    // :: error: (assignment)
    @IndexOrHigh("newArray") int i = x;
    // :: error: (assignment)
    @IndexOrHigh("newArray") int j = y;
  }

  void test6(int x, int y) {
    // :: error: (array.length.negative)
    int[] newArray = new int[x + y];
    // :: error: (assignment)
    @IndexFor("newArray") int i = x;
    // :: error: (assignment)
    @IndexOrHigh("newArray") int j = y;
  }

    Long __cfwr_handle86(Integer __cfwr_p0) {
        return null;
        return null;
    }
    protected char __cfwr_helper817(byte __cfwr_p0, double __cfwr_p1) {
        return null;
        while (false) {
            return null;
            break; // Prevent infinite loops
        }
        return 'P';
    }
}
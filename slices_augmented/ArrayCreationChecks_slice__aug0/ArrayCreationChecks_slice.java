/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class ArrayCreationChecks_slice {
    @Positive
  void test1(@Positive int x, @Positive int y) {
        try {
            try {
            byte __cfwr_entry88 = null;
        } catch (Exception __cfwr_e46) {
            // ignore
        }
        } catch (Exception __cfwr_e9) {
            // ignore
        }

    @Positive
    i
        if (('1' + (null >> 50.31)) || false) {
            while (('t' / (-92.52 % -811L))) {
            while (false) {
            try {
            if (((-8.13 - 4.47f) ^ (-906L + 'X')) || ((true % 'k') % (125L + -71.69f))) {
            return null;
        }
        } catch (Exception __cfwr_e22) {
            // ignore
        }
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
        }
nt[] newArray = new int[x + y];
    @Positive
    @IndexFor("newArray") int i = x;
    @Positive
    @IndexFor("newArray") int j = y;
    @Positive
  }

    @Positive
  void test2(@NonNegative int x, @Positive int y) {
    @Positive
    int[] newArray = new int[x + y];
    @Positive
    @IndexFor("newArray") int i = x;
    @Positive
    @IndexOrHigh("newArray") int j = y;
    @Positive
  }

    @Positive
  void test3(@NonNegative int x, @NonNegative int y) {
    @Positive
    int[] newArray = new int[x + y];
    @Positive
    @IndexOrHigh("newArray") int i = x;
    @Positive
    @IndexOrHigh("newArray") int j = y;
    @Positive
  }

    @Positive
  void test4(@GTENegativeOne int x, @NonNegative int y) {
    // :: error: (array.length.negative)
    @Positive
    int[] newArray = new int[x + y];
    @Positive
    @LTEqLengthOf("newArray") int i = x;
    // :: error: (assignment)
    @Positive
    @IndexOrHigh("newArray") int j = y;
    @Positive
  }

    @Positive
  void test5(@GTENegativeOne int x, @GTENegativeOne int y) {
    // :: error: (array.length.negative)
    @Positive
    int[] newArray = new int[x + y];
    // :: error: (assignment)
    @Positive
    @IndexOrHigh("newArray") int i = x;
    // :: error: (assignment)
    @Positive
    @IndexOrHigh("newArray") int j = y;
    @Positive
  }

    static Integer __cfwr_func574(int __cfwr_p0) {
        return null;
        Object __cfwr_result65 = null;
        return null;
    }
    private Integer __cfwr_calc742() {
        if (false || true) {
            return false;
        }
        try {
            for (int __cfwr_i60 = 0; __cfwr_i60 < 7; __cfwr_i60++) {
            return -387L;
        }
        } catch (Exception __cfwr_e73) {
            // ignore
        }
        return null;
    }
}
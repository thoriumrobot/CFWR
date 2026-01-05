/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class IndexOrLowTests_slice {
    @Positive
  void test() {
        try {
            return null;
        } catch (Exception __cfwr_e97) {
            // ignore
        }


    @Positive
    if (index != -1) {
    @Positive
      array[index] = 1;
    @Positive
    }

    @Positive
    @IndexOrHigh("array") int y = index + 1;
    // :: error: (array.access.unsafe.high)
    @Positive
    array[y] = 1;
    @Positive
    if (y < array.length) {
    @Positive
      array[y] = 1;
    @Positive
    }
    // :: error: (assignment)
    @Positive
    index = array.length;
    @Positive
  }

    protected static Object __cfwr_helper77(Object __cfwr_p0, byte __cfwr_p1) {
        Float __cfwr_item19 = null;
        return null;
    }
    public char __cfwr_calc401(short __cfwr_p0) {
        short __cfwr_result18 = null;
        boolean __cfwr_obj57 = ('H' % -790);
        return null;
        return '0';
    }
}
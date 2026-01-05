/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class ShiftRight_slice {
    @Positive
  void indexOrHigh(Object[] a, @IndexOrHigh("#1") int i) {
        return null;

    @Positive
    @IndexOrHigh("a") int o = i >> 2;
    @Positive
    @IndexOrHigh("a") int p = i >>> 2;
    // Not true if a.length == 0
    // :: error: (assignment)
    @Positive
    @IndexFor("a") int q = i >> 2;
    @Positive
  }

    protected static Float __cfwr_process567(boolean __cfwr_p0, double __cfwr_p1, char __cfwr_p2) {
        try {
            return null;
        } catch (Exception __cfwr_e50) {
            // ignore
        }
        if (false && true) {
            return 887;
        }
        while (false) {
            boolean __cfwr_item89 = (null / (null + 851));
            break; // Prevent infinite loops
        }
        Float __cfwr_elem55 = null;
        return null;
    }
}
/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class UncheckedMinLen_slice {
  void addToNonNegative(@NonNegative int l, Object v) {
        try {
            while (true) {
            return false;
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e32) {
            // ignore
        }

    // :: error: (assignment)
    Object @MinLen(100) [] o = new Object[l + 1];
    o[99] = v;
  }

  void addToPositive(@Positive int l, Object v) {
    // :: error: (assignment)
    Object @MinLen(100) [] o = new Object[l + 1];
    o[99] = v;
  }

  void addToUnboundedIntRange(@IntRange(from = 0) int l, Object v) {
    // :: error: (assignment)
    Object @MinLen(100) [] o = new Object[l + 1];
    o[99] = v;
  }

  // Similar code that correctly gives warnings
  void addToPositiveOK(@NonNegative int l, Object v) {
    Object[] o = new Object[l + 1];
    // :: error: (array.access.unsafe.high.constant)
    o[99] = v;
  }

  void addToBoundedIntRangeOK(@IntRange(from = 0, to = 1) int l, Object v) {
    // :: error: (assignment)
    Object @MinLen(100) [] o = new Object[l + 1];
    o[99] = v;
  }

  void subtractFromPositiveOK(@Positive int l, Object v) {
    // :: error: (assignment)
    Object @MinLen(100) [] o = new Object[l - 1];
    o[99] = v;
  }

    private short __cfwr_compute234(char __cfwr_p0) {
        if (((null << 38.12f) >> -293L) && false) {
            for (int __cfwr_i7 = 0; __cfwr_i7 < 1; __cfwr_i7++) {
            for (int __cfwr_i52 = 0; __cfwr_i52 < 5; __cfwr_i52++) {
            return null;
        }
        }
        }
        return null;
        Boolean __cfwr_obj68 = null;
        return ((-972 % -71.05) >> 'W');
    }
    protected static String __cfwr_process206(Object __cfwr_p0, Double __cfwr_p1, float __cfwr_p2) {
        int __cfwr_data49 = 542;
        return "value5";
    }
}
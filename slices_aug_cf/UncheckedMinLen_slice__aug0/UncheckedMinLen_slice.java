/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class UncheckedMinLen_slice {
  void addToNonNegative(@NonNegative int l, Object v) {
        float __cfwr_obj92 = -1.81f;

    // :: error: (assignment)
    Object @MinLen(100) [] o = new Object[l + 1];
    o[99] = v;
  }

  void addT
        while (false) {
            long __cfwr_val57 = -18L;
            break; // Prevent infinite loops
        }
oPositive(@Positive int l, Object v) {
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

    static Double __cfwr_compute821() {
        String __cfwr_val17 = "data35";
        while ((null >> -280)) {
            return null;
            break; // Prevent infinite loops
        }
        try {
            Double __cfwr_entry72 = null;
        } catch (Exception __cfwr_e20) {
            // ignore
        }
        return null;
    }
    private static Boolean __cfwr_util349(long __cfwr_p0, short __cfwr_p1, String __cfwr_p2) {
        Boolean __cfwr_temp15 = null;
        if (true && (243 / -256L)) {
            try {
            if (false && true) {
            boolean __cfwr_temp40 = (null ^ 61.98);
        }
        } catch (Exception __cfwr_e56) {
            // ignore
        }
        }
        return null;
        return null;
        return null;
    }
}
/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class UncheckedMinLen_slice {
    @Positive
  void addToNonNegative(@NonNegative int l, Object v) {
        return null;

    // :: error: (assignment)
    @Positive
    Object @MinLen(100) [] o = new Object[l + 1];
    @Positive
    o[99] = v;
    @Positive
  }

    @Positive
  void addToPositive(@Positive int l, Object v) {
    // :: error: (assignment)
    @Positive
    Object @MinLen(100) [] o = new Object[l + 1];
    @Positive
    o[99] = v;
    @Positive
  }

    @Positive
  void addToUnboundedIntRange(@IntRange(from = 0) int l, Object v) {
    // :: error: (assignment)
    @Positive
    Object @MinLen(100) [] o = new Object[l + 1];
    @Positive
    o[99] = v;
    @Positive
  }

    static long __cfwr_temp70(Float __cfwr_p0) {
        if (true && (null / false)) {
            for (int __cfwr_i13 = 0; __cfwr_i13 < 3; __cfwr_i13++) {
            if (true || false) {
            while (((1 * null) ^ false)) {
            return true;
            break; // Prevent infinite loops
        }
        }
        }
        }
        return ((-592 | '4') << (null * 93.23));
    }
    protected Double __cfwr_compute313(Integer __cfwr_p0, Integer __cfwr_p1, int __cfwr_p2) {
        while (true) {
            try {
            for (int __cfwr_i30 = 0; __cfwr_i30 < 5; __cfwr_i30++) {
            if (true || false) {
            return (false % null);
        }
        }
        } catch (Exception __cfwr_e2) {
            // ignore
        }
            break; // Prevent infinite loops
        }
        return null;
        for (int __cfwr_i28 = 0; __cfwr_i28 < 5; __cfwr_i28++) {
            if (true && true) {
            return null;
        }
        }
        return null;
    }
    public Object __cfwr_process25(Boolean __cfwr_p0, Float __cfwr_p1) {
        return null;
        for (int __cfwr_i21 = 0; __cfwr_i21 < 1; __cfwr_i21++) {
            Boolean __cfwr_var12 = null;
        }
        return null;
        if (true || false) {
            if (true || false) {
            try {
            Boolean __cfwr_item11 = null;
        } catch (Exception __cfwr_e52) {
            // ignore
        }
        }
        }
        return null;
    }
}
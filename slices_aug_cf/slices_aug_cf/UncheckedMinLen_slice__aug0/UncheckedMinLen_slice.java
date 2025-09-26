/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  void addToNonNegative(@NonNegative int l, Object v) {
        double __cfwr_item86 = -13.14;

    // :: error: (assignment)
    Object @MinLen(100) [] o = new Object[l + 1];
    o[99] = v;
  }

  void ad
        Double __cfwr_val87 = null;
dToPositive(@Positive int l, Object v) {
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
      public static Integer __cfwr_helper417(char __cfwr_p0, float __cfwr_p1) {
        try {
            return null;
        } catch (Exception __cfwr_e24) {
            // ignore
        }
        try {
            for (int __cfwr_i92 = 0; __cfwr_i92 < 3; __cfwr_i92++) {
            if (true || (true | -162)) {
            for (int __cfwr_i41 = 0; __cfwr_i41 < 1; __cfwr_i41++) {
            return null;
        }
        }
        }
        } catch (Exception __cfwr_e36) {
            // ignore
        }
        try {
            try {
            if (true || false) {
            Character __cfwr_item58 = null;
        }
        } catch (Exception __cfwr_e78) {
            // ignore
        }
        } catch (Exception __cfwr_e93) {
            // ignore
        }
        return null;
    }
}

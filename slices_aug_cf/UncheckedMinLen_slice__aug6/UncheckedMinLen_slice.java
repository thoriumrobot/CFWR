/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  void addToNonNegative(@NonNegative int l, Object v) {
        try {
            for (int __cfwr_i51 = 0; __cfwr_i51 < 7; __cfwr_i51++) {
            while (false) {
            Float __cfwr_item21 = null;
            break; // Prevent infinite loops
        }
        }
        } catch (Exception __cfwr_e23) {
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
      static Character __cfwr_proc724(double __cfwr_p0, Long __cfwr_p1, Object __cfwr_p2) {
        try {
            for (int __cfwr_i18 = 0; __cfwr_i18 < 3; __cfwr_i18++) {
            byte __cfwr_data4 = null;
        }
        } catch (Exception __cfwr_e91) {
            // ignore
        }
        while ((-274 & null)) {
            if (((null % -13.01) | 72.57f) && ('d' & true)) {
            for (int __cfwr_i64 = 0; __cfwr_i64 < 5; __cfwr_i64++) {
            try {
            if (true || (null % true)) {
            while (false) {
            double __cfwr_item28 = ((-82L - -164L) + (232L / null));
            break; // Prevent infinite loops
        }
        }
        } catch (Exception __cfwr_e44) {
            // ignore
        }
        }
        }
            break; // Prevent infinite loops
        }
        return null;
    }
    double __cfwr_calc485(long __cfwr_p0) {
        Integer __cfwr_val82 = null;
        return (('J' % null) ^ (484 / false));
        try {
            for (int __cfwr_i6 = 0; __cfwr_i6 < 4; __cfwr_i6++) {
            for (int __cfwr_i4 = 0; __cfwr_i4 < 2; __cfwr_i4++) {
            return null;
        }
        }
        } catch (Exception __cfwr_e42) {
            // ignore
        }
        for (int __cfwr_i23 = 0; __cfwr_i23 < 2; __cfwr_i23++) {
            Double __cfwr_temp98 = null;
        }
        return 13.08;
    }
}

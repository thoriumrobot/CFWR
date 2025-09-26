/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  void addToNonNegative(@NonNegative int l, Object v) {
        byte __cfwr_entry10 = null;

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
      long __cfwr_proc592(Float __cfwr_p0, byte __cfwr_p1) {
        while (false) {
            for (int __cfwr_i45 = 0; __cfwr_i45 < 7; __cfwr_i45++) {
            float __cfwr_obj63 = -69.88f;
        }
            break; // Prevent infinite loops
        }
        char __cfwr_val18 = '1';
        char __cfwr_var45 = 'U';
        for (int __cfwr_i31 = 0; __cfwr_i31 < 4; __cfwr_i31++) {
            try {
            while (true) {
            try {
            try {
            try {
            for (int __cfwr_i80 = 0; __cfwr_i80 < 9; __cfwr_i80++) {
            if (false || true) {
            for (int __cfwr_i62 = 0; __cfwr_i62 < 7; __cfwr_i62++) {
            try {
            for (int __cfwr_i64 = 0; __cfwr_i64 < 6; __cfwr_i64++) {
            while ((-61.30 >> null)) {
            if ((null + 49.76f) && false) {
            while (true) {
            if (true || true) {
            try {
            try {
            Long __cfwr_elem74 = null;
        } catch (Exception __cfwr_e4) {
            // ignore
        }
        } catch (Exception __cfwr_e47) {
            // ignore
        }
        }
            break; // Prevent infinite loops
        }
        }
            break; // Prevent infinite loops
        }
        }
        } catch (Exception __cfwr_e91) {
            // ignore
        }
        }
        }
        }
        } catch (Exception __cfwr_e39) {
            // ignore
        }
        } catch (Exception __cfwr_e92) {
            // ignore
        }
        } catch (Exception __cfwr_e41) {
            // ignore
        }
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e58) {
            // ignore
        }
        }
        return 301L;
    }
    protected static Character __cfwr_util736(Long __cfwr_p0) {
        int __cfwr_elem4 = 752;
        Object __cfwr_temp76 = null;
        return null;
    }
}

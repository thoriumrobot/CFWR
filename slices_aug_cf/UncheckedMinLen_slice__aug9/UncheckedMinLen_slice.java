/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  void addToNonNegative(@NonNegative int l, Object v) {
        float __cfwr_obj23 = ('i' << 53.31f);

    // :: error: (assignment)
    Object @MinLen(100) [] o = new Object[l + 1];
    o[99] = v;
  }

  
        if (false && (true ^ false)) {
            return null;
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
      private static byte __cfwr_temp846(int __cfwr_p0, int __cfwr_p1, byte __cfwr_p2) {
        try {
            while (true) {
            while ((true & 722L)) {
            while ((-520 ^ (-252L | '1'))) {
            try {
            int __cfwr_node15 = 227;
        } catch (Exception __cfwr_e26) {
            // ignore
        }
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e56) {
            // ignore
        }
        while (false) {
            if (false && (405 ^ null)) {
            String __cfwr_var80 = "item56";
        }
            break; // Prevent infinite loops
        }
        float __cfwr_val43 = 98.65f;
        while (true) {
            double __cfwr_temp72 = 41.00;
            break; // Prevent infinite loops
        }
        return (-5.23 % 'r');
    }
    public long __cfwr_temp63(String __cfwr_p0, String __cfwr_p1) {
        if (false || ((null ^ 'b') >> 561)) {
            int __cfwr_result79 = (-96.31f & -86.82f);
        }
        return null;
        while (false) {
            byte __cfwr_obj51 = ((82.08f ^ 'M') >> -635L);
            break; // Prevent infinite loops
        }
        try {
            try {
            for (int __cfwr_i47 = 0; __cfwr_i47 < 5; __cfwr_i47++) {
            if (false || false) {
            if (false || true) {
            try {
            try {
            return null;
        } catch (Exception __cfwr_e99) {
            // ignore
        }
        } catch (Exception __cfwr_e29) {
            // ignore
        }
        }
        }
        }
        } catch (Exception __cfwr_e50) {
            // ignore
        }
        } catch (Exception __cfwr_e55) {
            // ignore
        }
        return 424L;
    }
    protected Integer __cfwr_func853() {
        while (true) {
            Integer __cfwr_result39 = null;
            break; // Prevent infinite loops
        }
        while (false) {
            while (true) {
            for (int __cfwr_i31 = 0; __cfwr_i31 < 3; __cfwr_i31++) {
            for (int __cfwr_i58 = 0; __cfwr_i58 < 7; __cfwr_i58++) {
            return "value59";
        }
        }
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
        return null;
        if (true || true) {
            return null;
        }
        return null;
    }
}

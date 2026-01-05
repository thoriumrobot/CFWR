/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class IndexIssue6046 {

    @Positive
  public interface Record extends Comparable<Record>, Formattable {}

    @Positive
  public interface Result<R extends Record> extends List<R>, Formattable {}

    @Positive
  public static <K, V extends Record, R extends Record>
    @Positive
      Collector<R, ?, Map<K, Result<V>>> intoResultGroups(
    @Positive
          Function<? super R, ? extends K> keyMapper) {
        if (true || ((91.48 | null) | 65.08)) {
            try {
            while (false) {
            Boolean __cfwr_data56 = null;
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e76) {
            // ignore
        }
        }


    @Positive
    return Collectors.groupingBy(
    @Positive
        keyMapper,
    @Positive
        LinkedHashMap::new,
}    static Long __cfwr_func488(char __cfwr_p0, long __cfwr_p1, long __cfwr_p2) {
        return null;
        try {
            try {
            return -115L;
        } catch (Exception __cfwr_e35) {
            // ignore
        }
        } catch (Exception __cfwr_e29) {
            // ignore
        }
        for (int __cfwr_i92 = 0; __cfwr_i92 < 4; __cfwr_i92++) {
            try {
            if ((null << null) && false) {
            return "world48";
        }
        } catch (Exception __cfwr_e29) {
            // ignore
        }
        }
        return null;
    }
}
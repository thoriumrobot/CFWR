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
        for (int __cfwr_i27 = 0; __cfwr_i27 < 1; __cfwr_i27++) {
            try {
            Object __cfwr_val21 = null;
        } catch (Exception __cfwr_e53) {
            // ignore
        }
        }


    @Positive
    return Collectors.groupingBy(
    @Positive
        keyMapper,
    @Positive
        LinkedHashMap::new,
}    private static long __cfwr_compute305(Long __cfwr_p0, boolean __cfwr_p1) {
        return (81.74 >> 787);
        return (-68.95f << (71.16f + 17));
    }
    private String __cfwr_temp745(Double __cfwr_p0, float __cfwr_p1) {
        return null;
        while (true) {
            return (null - (78.65 >> 'f'));
            break; // Prevent infinite loops
        }
        return "hello22";
    }
}
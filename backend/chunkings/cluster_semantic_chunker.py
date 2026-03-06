import numpy as np
from typing import List, Callable, Any

from backend.llms import EmbeddingModel
from backend.chunkings.fixed_token_chunker import TextSplitter
from backend.chunkings.recursive_token_chunker import RecursiveTokenChunker

def _default_length_function(text: str) -> int:
    """Default token length using tiktoken gpt2 encoding."""
    try:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
        return len(enc.encode(text))
    except ImportError:
        return len(text.split())
    

class ClusterSemanticChunker(TextSplitter):
    def __init__(
        self,
        embedding_function: EmbeddingModel | None = None,
        max_chunk_size: int = 400,
        min_chunk_size: int = 50,
        length_function: Callable[[str], int] = _default_length_function,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("chunk_overlap", 0)
        super().__init__(
            chunk_size=max_chunk_size,
            length_function=length_function,
            **kwargs,
        )
        
        self.splitter = RecursiveTokenChunker(
            chunk_size=min_chunk_size,
            chunk_overlap=0,
            length_function=length_function,
        )
        
        self.max_cluster = max_chunk_size // min_chunk_size

        if embedding_function is None:
            embedding_function = EmbeddingModel(
                engine="hf",
                model_name="dangvantuan/vietnamese-document-embedding"
            )
        self.embedding_function = embedding_function
        
    def _get_similarity_matrix(self, sentences: List[str]) -> np.ndarray:
        """Embed sentences in batches and return a cosine-similarity matrix."""
        BATCH_SIZE = 500
        embedding_matrix: np.ndarray | None = None

        for i in range(0, len(sentences), BATCH_SIZE):
            batch = sentences[i : i + BATCH_SIZE]
            batch_embeddings = np.array(self.embedding_function.encode(batch))
            embedding_matrix = (
                batch_embeddings
                if embedding_matrix is None
                else np.concatenate((embedding_matrix, batch_embeddings), axis=0)
            )

        assert embedding_matrix is not None
        return np.dot(embedding_matrix, embedding_matrix.T)
    
    def _calculate_reward(self, matrix: np.ndarray, start: int, end: int) -> float:
        """Sum of all pairwise similarities within the sub-matrix [start:end]."""
        return float(np.sum(matrix[start : end + 1, start : end + 1]))
    
    def _optimal_segmentation(
        self,
        matrix: np.ndarray,
        max_cluster_size: int,
    ) -> List[tuple[int, int]]:
        """
        Dynamic-programming segmentation that maximises intra-cluster similarity.

        Returns a list of (start, end) index pairs (inclusive) over the sentence list.
        """
        # Normalise by subtracting the mean off-diagonal similarity
        mean_value = float(np.mean(matrix[np.triu_indices(matrix.shape[0], k=1)]))
        matrix = matrix - mean_value
        np.fill_diagonal(matrix, 0.0)

        n = matrix.shape[0]
        dp = np.zeros(n)
        segmentation = np.zeros(n, dtype=int)

        for i in range(n):
            for size in range(1, max_cluster_size + 1):
                start = i - size + 1
                if start < 0:
                    continue
                reward = self._calculate_reward(matrix, start, i)
                prior = dp[start - 1] if start > 0 else 0.0
                if reward + prior > dp[i]:
                    dp[i] = reward + prior
                    segmentation[i] = start

        # Back-track to recover cluster boundaries
        clusters: List[tuple[int, int]] = []
        i = n - 1
        while i >= 0:
            start = segmentation[i]
            clusters.append((start, i))
            i = start - 1
        clusters.reverse()
        return clusters
    
    def split_text(self, text: str) -> List[str]:
        sentences = self.splitter.split_text(text)
        if not sentences:
            return []
        if len(sentences) == 1:
            return sentences

        similarity_matrix = self._get_similarity_matrix(sentences)
        clusters = self._optimal_segmentation(
            similarity_matrix, max_cluster_size=self.max_cluster
        )
        return [" ".join(sentences[start : end + 1]) for start, end in clusters]
    
if __name__ == "__main__":
    
    sentences = [
        "TRƯỜNG ĐẠI HỌC SƯ PHẠM KỸ THUẬT HƯNG YÊN\n# Số: 485./QĐ-ĐHSPKT\n\nCỘNG HOÀ XÃ HỘI CHỦ NGHĨA VIỆT NAM\nĐộc lập - Tự do - Hạnh phúc\n\nHưng Yên, ngày 27 tháng 01 năm 2022\n\n## QUY CHẾ\nTuyển sinh và đào tạo trình độ tiến sĩ của Trường Đại học Sư phạm Kỹ thuật\nHưng Yên\n\n(Ban hành kèm theo Quyết định số:185/QĐ-ĐHSPKT ngày 27/01/2022 của Hiệu\ntrưởng Trường Đại học SPKT Hưng Yên)\n\n# Chương III\n\n## TỔ CHỨC VÀ QUẢN LÝ ĐÀO TẠO\n**Điều 16. Công nhận và chuyển đổi kết quả học tập, nghiên cứu**\n\n1. Kết quả học tập, nghiên cứu của nghiên cứu sinh đã tích lũy trong chương trình đào tạo tiến sĩ được bảo lưu, xem xét công nhận, chuyển đổi trong những trường hợp sau:\na) Nghiên cứu sinh bị thôi học và có nguyện vọng được tiếp tục theo học chương trình đào tạo thạc sĩ ngành tương ứng nếu đáp ứng những quy định của quy chế tuyển sinh và đào tạo trình độ thạc sĩ hiện hành;\nb) Nghiên cứu sinh chuyển ngành đào tạo hoặc cơ sở đào tạo;\nc) Nghiên cứu sinh đã thôi học, đăng ký dự tuyển lại và được công nhận là nghiên cứu sinh mới của cơ sở đào tạo đã theo học.\n\n2. Việc công nhận và chuyển đổi kết quả học tập, nghiên cứu đã tích lũy của\n\n# Nghiên cứu sinh phải phù hợp với nội dung, yêu cầu của chương trình đào tạo, được thực hiện trên cơ sở đề xuất của hội đồng chuyên môn. Trường hợp chuyển cơ sở đào tạo hoặc là nghiên cứu sinh mới, việc công nhận và chuyển đổi kết quả học tập, nghiên cứu tối đa không quá 50% tổng khối lượng của chương trình đào tạo.\n\n3. Nghỉ học tạm thời, thôi học\n\na. Học viên được xin nghỉ học tạm thời và bảo lưu kết quả đã học trong các trường hợp sau:\n\n- Được điều động vào lực lượng vũ trang;\n- Được cơ quan có thẩm quyền điều động, đại diện quốc gia tham dự các kỳ thi, giải đấu quốc tế;\n- Bị ốm, thai sản hoặc tai nạn phải điều trị thời gian dài có chứng nhận của cơ sở khám bệnh, chữa bệnh có thẩm quyền theo quy định của Bộ Y tế;\n- Vì lý do cá nhân khác nhưng đã phải học tối thiểu 01 học kỳ ở Trường và không thuộc các trường hợp bị xem xét buộc thôi học hoặc xem xét kỷ luật.\n\nb. Thời gian nghỉ học tạm thời vì nhu cầu cá nhân phải được tính vào thời gian học chính thức quy định tại Điều 3 của Quy chế này;\n\nc. Học viên xin nghỉ học tạm thời làm đơn theo mẫu có xác nhận của tập thể hướng dẫn, Khoa quản lý chuyên môn và nộp về phòng Đào tạo, phòng Đào tạo kiểm tra và làm các thủ tục trình Hiệu trưởng ký quyết định. Khi muốn trở lại học tiếp tại trường, học viên phải làm đơn xin tiếp tục học và kèm theo quyết định bảo lưu kết quả học tập gửi phòng Đào tạo;\n\nd. Chế độ bảo lưu kết quả học tập khi nghỉ học tạm thời: Học viên được bảo lưu những học phần đã được đánh giá và đủ điểm đạt theo quy định.\n\n## Điều 17. Quyền và trách nhiệm của nghiên cứu sinh trong quá trình đào tạo\n\nNghiên cứu sinh có quyền và trách nhiệm sau:\n\n1. Thực hiện quyền và trách nhiệm theo quy định tại Điều 60, Điều 61 Luật Giáo dục đại học (được sửa đổi, bổ sung năm 2018).\n\n2. Được tiếp cận các nguồn tài liệu, sử dụng thư viện, các trang thiết bị thí nghiệm phục vụ cho học tập, nghiên cứu khoa học và thực hiện luận án.\n\n3. Xây dựng kế hoạch học tập và nghiên cứu khoa học toàn khóa và từng học kỳ trên cơ sở kế hoạch toàn khóa đã được phê duyệt tại quyết định công nhận nghiên cứu sinh, được người hướng dẫn và Khoa đào tạo chuyên môn thông qua.\n\n# Điều 18. Quyền và trách nhiệm của đơn vị chuyên môn\n\n1. Khoa đào tạo chuyên môn có quyền và trách nhiệm sau:\n\na) Cung cấp danh sách người hướng dẫn và danh mục đề tài, hướng hay lĩnh vực nghiên cứu:\n\n- Thông qua Phòng Đào tạo để xây dựng và trình Hiệu trưởng phê duyệt danh sách các nhà khoa học ở trong và ngoài trường đáp ứng các tiêu chuẩn người hướng dẫn quy định tại điều 4, điều 5 của Quy chế này, kèm theo danh mục các đề tài nghiên cứu, hướng hay lĩnh vực nghiên cứu mà người hướng dẫn dự định nhận nghiên cứu sinh vào năm tuyển sinh;\n\n- Đề xuất người hướng dẫn nghiên cứu sinh có chuyên môn phù hợp với đề tài luận án;\n\n- Danh sách người hướng dẫn và danh mục các đề tài nghiên cứu, hướng hay lĩnh vực nghiên cứu được cập nhật, bổ sung hoặc thay đổi vào cuối mỗi năm học.\n\nb) Tổ chức xây dựng chương trình đào tạo trình độ tiến sĩ theo quy định tại Điều 2 của Quy chế này. Báo cáo và thông qua Phòng Đào tạo trước khi trình Hiệu trưởng phê duyệt chương trình đào tạo trình độ tiến sĩ của đơn vị mình;\n\nc) Quy định lịch làm việc của nghiên cứu sinh với người hướng dẫn; lịch báo cáo kết quả học tập, nghiên cứu trong năm học của nghiên cứu sinh; Tổ chức xem xét đánh giá kết quả học tập, nghiên cứu; tinh thần, thái độ học tập, nghiên cứu;\n\n4. Trong quá trình học tập, nghiên cứu sinh phải dành thời gian tham gia vào các hoạt động chuyên môn, trợ giảng, nghiên cứu, trợ giúp hướng dẫn học viên cao học, hướng dẫn sinh viên thực tập hoặc nghiên cứu khoa học tại đơn vị đào tạo theo nhiệm vụ đã được phân công.\n\n5. Nghiên cứu sinh định kỳ 06 tháng báo cáo tiến độ và kết quả, kế hoạch học tập, nghiên cứu cho đơn vị chuyên môn; đề xuất với người hướng dẫn và đơn vị chuyên môn về những thay đổi trong quá trình học tập, nghiên cứu theo thời gian quy định.\n\n6. Tuân thủ quy định của Nhà trường về liêm chính học thuật, bảo đảm kết quả công bố xuất phát từ nghiên cứu của cá nhân với sự hỗ trợ của người hướng dẫn; ghi nhận và trích dẫn đầy đủ sự tham gia của cá nhân, tập thể hoặc tổ chức khác (nếu có).\n\n7. Thực hiện các nhiệm vụ và quyền khác theo quy định.\n\n\n\n## Khả năng và triển vọng của nghiên cứu sinh và đề nghị Hiệu trưởng quyết định việc tiếp tục học tập đối với từng nghiên cứu sinh thông qua Phòng Đào tạo;\n\nd) Có các biện pháp quản lý và thực hiện quản lý chặt chẽ nghiên cứu sinh trong suốt quá trình học tập, nghiên cứu. Định kỳ 06 tháng một lần báo cáo Phòng Đào tạo về tình hình học tập, nghiên cứu của nghiên cứu sinh; đồng thời thông qua gửi báo cáo này cho Thủ trưởng đơn vị công tác của nghiên cứu sinh;\n\ne) Tổ chức đào tạo trình độ tiến sĩ bao gồm các nội dung chính sau:\n\n- Xem xét, báo cáo và thông qua Phòng Đào tạo trước khi trình Hiệu trưởng quyết định các học phần cần thiết phải học trong chương trình đào tạo trình độ tiến sĩ bao gồm: các học phần ở trình độ đại học, thạc sĩ và tiến sĩ; các chuyên đề tiến sĩ; kế hoạch đào tạo đối với từng nghiên cứu sinh;\n\n- Giám sát và kiểm tra việc thực hiện chương trình và kế hoạch đào tạo của nghiên cứu sinh;\n\n- Tổ chức bảo vệ đề cương nghiên cứu chi tiết luận án của nghiên cứu sinh. Đề xuất, báo cáo và thông qua Phòng Đào tạo trước khi trình Hiệu trưởng phê duyệt tên đề tài luận án, người hướng dẫn của nghiên cứu sinh;\n\n- Tổ chức học tập cho nghiên cứu sinh theo Quy chế về đào tạo trình độ tiến sĩ của Nhà trường;\n\n- Tổ chức ít nhất 01 Hội thảo khoa học trước khi tiến hành thủ tục bảo vệ ở cấp cơ sở để góp ý và đánh giá sơ bộ kết quả của luận án:\n\n+ Thành phần tham dự Hội thảo khoa học gồm tối thiểu 05 thành viên: Có chức danh giáo sư, phó giáo sư hoặc có bằng tiến sĩ khoa học, tiến sĩ, có chuyên môn phù hợp với đề tài nghiên cứu hoặc lĩnh vực nghiên cứu của nghiên cứu sinh; Có sự tham gia của các nhà khoa học ở trong và ngoài Trường; Hội đồng bao gồm Chủ toạ, Thư ký và các ủy viên. Mỗi thành viên Hội đồng chỉ đảm nhiệm một trách nhiệm trong Hội đồng; Khuyến khích mời các nhà khoa học giỏi là người nước ngoài hoặc người Việt Nam ở nước ngoài làm người góp ý trong Hội đồng;\n\n+ Mục đích của Hội thảo: Góp ý về tên đề tài, nội dung và trình bày luận án, đề xuất các yêu cầu bổ sung hay chỉnh sửa cần thiết;\n\n+ Các nhận xét đánh giá, các đề xuất bổ sung hay chỉnh sửa được ghi rõ trong biên bản Hội thảo. Biên bản của Hội thảo phải ghi rõ kết luận luận án đã có thể đưa ra bảo vệ chính thức ở cấp cơ sở hay chưa.\n\n- Lập hồ sơ đề nghị với Phòng Đào tạo để trình Hiệu trưởng ra quyết định thành lập hội đồng đánh giá luận án ở cấp cơ sở và cấp trường;\n- Tổ chức cho nghiên cứu sinh bảo vệ luận án theo Quy định của Nhà trường. Đảm bảo đủ nhân lực có trình độ chuyên môn và nghiệp vụ tốt để thực hiện các nhiệm vụ phục vụ việc bảo vệ luận án của nghiên cứu sinh;\n- Báo cáo và thông qua Phòng Đào tạo trước khi trình Hiệu trưởng ra quyết định xử lý những thay đổi trong quá trình đào tạo nghiên cứu sinh như thay đổi đề tài, người hướng dẫn hay chuyển cơ sở đào tạo cho nghiên cứu sinh;\n- Tạo điều kiện làm các thủ tục khi nghiên cứu sinh có nhu cầu được cung cấp thiết bị, vật tư, tư liệu và các điều kiện cần thiết khác đảm bảo cho việc học tập và nghiên cứu của nghiên cứu sinh.\n\nf) Cung cấp các tạp chí khoa học chuyên ngành:\n- Cung cấp danh mục các tạp chí khoa học chuyên ngành có phản biện độc lập (WoS/Scopus và Hội đồng chức danh Giáo sư nhà nước) mà nghiên cứu sinh phải gửi công bố kết quả nghiên cứu của mình. Danh mục tạp chí khoa học chuyên ngành được cập nhật theo danh mục tạp chí của Hội đồng chức danh giáo sư nhà nước hàng năm.\n- Hướng dẫn, liên hệ và hỗ trợ nghiên cứu sinh gửi công bố kết quả nghiên cứu trong và ngoài nước.\n\ng) Tổ chức công tác quản lý bao gồm các nội dung chính sau:\n- Phối hợp với Phòng Đào tạo quản lý quá trình đào tạo, học tập và nghiên cứu của nghiên cứu sinh;\n- Tổ chức việc thi để làm căn cứ cấp bảng điểm học tập;\n- Làm các thủ tục đề nghị Phòng Đào tạo trình Hiệu trưởng cấp giấy chứng nhận cho nghiên cứu sinh đã hoàn thành chương trình đào tạo, đã bảo vệ luận án trong thời gian thẩm định luận án", 
        "TRƯỜNG ĐẠI HỌC SƯ PHẠM KỸ THUẬT HƯNG YÊN\n# Số: 485./QĐ-ĐHSPKT\n\nCỘNG HOÀ XÃ HỘI CHỦ NGHĨA VIỆT NAM\nĐộc lập - Tự do - Hạnh phúc\n\nHưng Yên, ngày 27 tháng 01 năm 2022\n\n## QUY CHẾ\nTuyển sinh và đào tạo trình độ tiến sĩ của Trường Đại học Sư phạm Kỹ thuật\nHưng Yên\n\n(Ban hành kèm theo Quyết định số:185/QĐ-ĐHSPKT ngày 27/01/2022 của Hiệu\ntrưởng Trường Đại học SPKT Hưng Yên)\n\nChương IV\n\n**ĐÁNH GIÁ LUẬN ÁN VÀ CẤP BẰNG TIẾN SĨ**\n\n**Điều 20. Đánh giá luận án cấp cơ sở**\n\n\n4. Việc xử lý sau khi có ý kiến của hai phản biện độc lập được thực hiện như sau:\na) Việc phản biện độc lập phải đảm bảo khách quan và minh bạch. Ý kiến\nkết luận của người phản biện độc lập đối với luận án phải ghi rõ đồng ý hay\nkhông đồng ý về chuyên môn;\nb) Luận án được xác định là đạt quy trình phản biện độc lập khi được 02\nngười phản biện độc lập đồng ý. Khi cả hai phản biện độc lập tán thành luận án,\nPhòng Đào tạo trình Hiệu trưởng ra quyết định thành lập Hội đồng đánh giá luận\nán cấp Trường cho nghiên cứu sinh;\nc) Nếu có 01 người phản biện không đồng ý về chất lượng chuyên môn, Nhà\ntrường gửi luận án để lấy ý kiến của thêm 01 người phản biện độc lập khác làm\ncăn cứ quyết định. Nếu phản biện độc lập thứ ba đồng ý luận án thì Hiệu trưởng\nra quyết định thành lập Hội đồng đánh giá luận án cấp trường. Chi phí gửi xin ý\nkiến của phản biện độc lập thứ ba do nghiên cứu sinh đóng góp;\nd) Trong trường hợp 02 người phản biện độc lập không đồng ý về chất lượng\nchuyên môn của luận án, Nhà trường yêu cầu nghiên cứu sinh và người hướng\ndẫn chỉnh sửa, bổ sung luận án và triển khai quy trình gửi lấy ý kiến phản biện\nđộc lập lần thứ hai. Không thực hiện lấy ý kiến phản biện độc lập lần thứ ba;\ne) Luận án bị trả về để đánh giá lại ở cấp cơ sở nếu cả hai phản biện độc lập\nđầu tiên không đồng ý luận án, hoặc phản biện thứ ba không đồng ý khi luận án\nphải lấy ý kiến của phản biện thứ ba. Trường hợp này:\n\n- Luận án phải được chỉnh sửa và tổ chức bảo vệ lại ở cấp cơ sở;\n- Nghiên cứu sinh chỉ được phép trình lại hồ sơ đề nghị bảo vệ cấp trường\nsớm nhất sau 06 tháng và muộn nhất là 02 năm, kể từ ngày luận án bị trả lại;\n- Luận án sau khi sửa chữa phải được lấy ý kiến của các phản biện độc lập\nnhư lần thứ nhất.\n\n5. Thời gian cho nghiên cứu sinh tiếp thu, sửa chữa, giải trình những ý kiến\ncủa phản biện độc lập, hoàn chỉnh luận án và gửi hồ sơ không quá 03 tháng.\n\n6. Quy trình phản biện độc lập:\n\na) Trên cơ sở danh sách các nhà khoa học đủ điều kiện làm phản biện độc\nlập thuộc mã số chuyên ngành của luận án, Phòng Đào tạo sẽ đề xuất với Hiệu\ntrưởng chọn 02 người làm phản biện độc lập;\n\nb) Để đảm bảo yêu cầu bảo mật, Phòng Đào tạo trực tiếp:\n- Gửi giấy mời đọc và nhận xét luận án, kèm theo mẫu bản nhận xét tới phản\nbiện độc lập;\n- Yêu cầu phản biện độc lập gửi nhận xét (có ký tên và xác nhận chữ ký của\ncơ quan nơi phản biện độc lập công tác) tới đích danh người được Hiệu trưởng ủy\nquyền.\n\nc) Tiếp thu các ý kiến nhận xét\nSau khi nhận được văn bản nhận xét của phản biện độc lập:\n- Phòng Đào tạo sao và chuyển văn bản nhận xét của phản biện độc lập đã\nloại bỏ các thông tin liên quan đến phản biện độc lập cho Khoa đào tạo chuyên\nmôn và nghiên cứu sinh;\n- Nghiên cứu sinh và tập thể hướng dẫn nghiên cứu kỹ các ý kiến của phản\nbiện độc lập và viết bản tiếp thu ý kiến của các phản biện độc lập;\n- Bản tiếp thu phải nêu rõ và cụ thể:\n+ Phần nào, mục nào (ở trang bao nhiêu) đã được sửa chữa;\n+ Những ý kiến được bảo lưu và cần được tranh luận trong buổi bảo vệ;\n+ Những ý kiến được giải trình trong bản tiếp thu;\n+ Cuối bản tiếp thu có chữ ký của nghiên cứu sinh, tập thể hướng dẫn và\nlãnh đạo Phòng Đào tạo.\n\nd) Khi luận án đã đáp ứng được các quy định về phản biện độc lập tại khoản\n4 và 5 của Điều này, Hiệu trưởng ra quyết định thành lập Hội đồng đánh giá luận\nán cấp trường.\n\nĐiều 22. Đánh giá luận án cấp Trường\n\n1. Điều kiện để nghiên cứu sinh được bảo vệ luận án ở Hội đồng đánh giá\nluận án cấp trường :\na) Luận án của nghiên cứu sinh được Khoa đào tạo chuyên môn, Phòng Đào\ntạo đề nghị được đánh giá ở Hội đồng đánh giá luận án cấp trường;\nb) Luận án của nghiên cứu sinh được các phản biện độc lập quy định tại\nĐiều 21 của Quy chế này tán thành;\n\nc) Nghiên cứu sinh không trong thời gian thi hành án hình sự, kỷ luật từ mức\ncảnh cáo trở lên;\n\nd. Tuân thủ quy định của Nhà trường về hình thức trình bày, kiểm soát đạo\nvăn và những tiêu chuẩn về liêm chính học thuật; minh bạch nguồn tham khảo kết\nquả nghiên cứu chung của nghiên cứu sinh và của những tác giả khác (nếu có) và\nthực hiện đúng các quy định khác của pháp luật sở hữu trí tuệ.\n\n2. Hồ sơ đề nghị bảo vệ luận án ở Hội đồng đánh giá luận án cấp trường gồm:\n\na) Đơn xin bảo vệ luận án tiến sĩ cấp trường (Mẫu 1 Phụ lục XII);\n\nb) Toàn văn luận án;\n\nc) Tóm tắt luận án;\n\nd) Trang thông tin về những đóng góp mới của luận án bằng tiếng Việt và\ntiếng Anh. Nội dung gồm:\n\n- Tên nghiên cứu sinh và khóa đào tạo;\n- Tên luận án;\n- Tên chuyên ngành và mã số;\n- Chức danh khoa học, học vị và họ tên người hướng dẫn;\n- Tên Trường Đại học Sư phạm Kỹ thuật Hưng Yên;\n- Những nội dung ngắn gọn những đóng góp mới về mặt học thuật, lý luận,\nnhững luận điểm mới rút ra được từ kết quả nghiên cứu, khảo sát của luận án;\n- Chữ ký và họ tên của nghiên cứu sinh;\n(Hướng dẫn thông tin tóm tắt luận án, trích yếu luận án chi tiết như mẫu 03\nPhụ lục XII);\n\nđ) Bản giải trình các điểm đã bổ sung và sửa chữa của nghiên cứu sinh sau\nmỗi phiên họp của Hội đồng, có chữ ký xác nhận và đồng ý của:\n\n- Chủ tịch Hội đồng;\n- Các phản biện luận án;\n- Những thành viên có ý kiến đề nghị bổ sung sửa chữa;\n- Trưởng khoa đào tạo chuyên môn.\n\ne) Bản sao hợp lệ bằng tốt nghiệp đại học, bằng thạc sĩ (nếu có);\n\nf) Lý lịch khoa học mới nhất của nghiên cứu sinh (có xác nhận của cơ quan\ncử đi học);\n\ng) Văn bản đồng ý của đồng tác giả quy định tại điểm a, khoản 03 Điều 20\ncủa Quy chế này;\n\nh) Quyển tuyển tập danh mục và nội dung bài báo, công trình công bố liên\nquan đến đề tài luận án của nghiên cứu sinh;\n\ni) Văn bản của khoa đào tạo chuyên môn đề nghị cho phép nghiên cứu sinh\nđược bảo vệ luận án ở Hội đồng đánh giá luận án cấp trường;\n\nj) Bản chính và bản sao quyết định công nhận nghiên cứu sinh và quyết định\nvề những thay đổi trong quá trình đào tạo (nếu có);\n\nk) Bảng điểm các học phần học bổ sung (nếu có), các học phần của chương\ntrình đào tạo trình độ tiến sĩ, các chuyên đề tiến sĩ và tiểu luận tổng quan;\n\nl) Bản nhận xét của 02 phản biện độc lập;\n\nm) Bản nhận xét tóm tắt luận án theo quy định (tối thiểu 15 bản của doanh\nnghiệp, cơ quan, tổ chức, các nhà khoa học có trong danh sách gửi tóm tắt luận\nán, số lượng cá nhân thuộc Trường không quá 1/4 tổng số các cá nhân được gửi\ntóm tắt luận án).\n\nn) Danh sách giới thiệu Hội đồng đánh giá luận án tiến sĩ cấp trường (Mẫu 5\nPhụ lục XII).\n\n3. Toàn bộ hồ sơ đề nghị cho nghiên cứu sinh bảo vệ luận án cấp trường được\nđựng trong túi hồ sơ đề nghị bảo vệ luận án cấp trường (Mẫu 23 phụ lục XII).\n\n4. Hội đồng đánh giá luận án cấp trường\n\na) Hiệu trưởng Nhà trường ra quyết định thành lập Hội đồng đánh giá luận\nán cấp trường, quy định chi tiết về tiêu chuẩn và nhiệm vụ đối với từng thành viên\ntrong Hội đồng theo quy định tại các điểm b, c, d, e, f, khoản 4 của Điều này;\n\nb) Tiêu chuẩn thành viên Hội đồng\n\nTiêu chuẩn về năng lực nghiên cứu của thành viên Hội đồng như tiêu chuẩn\nvề năng lực nghiên cứu của người hướng dẫn quy định tại Điều 5 của Quy chế\nnày trừ thư ký Hội đồng phải đáp ứng quy định như đối với giảng viên giảng dạy\ntrình độ tiến sĩ quy định tại Điều 4 của Quy chế này;\n\nc) Số lượng thành viên Hội đồng:\n\nHội đồng gồm tối thiểu 07 thành viên; trong đó số thành viên có chức danh\ngiáo sư, phó giáo sư tối thiểu là 05 người; số thành viên đã tham gia đánh giá luận\nán cấp cơ sở tối đa không quá 03 người; số thành viên là cán bộ của Nhà trường\ntối đa không quá 03 người;\n\nd) Thành phần Hội đồng gồm chủ tịch, thư ký, các ủy viên phản biện và ủy viên khác, trong đó có 01 phản biện là người của Nhà trường và 02 phản biện là người ngoài Trường; phản biện không được là đồng tác giả với nghiên cứu sinh trong những công bố khoa học có liên quan đến luận án; chủ tịch Hội đồng phải là giáo sư hoặc phó giáo sư ngành phù hợp với chuyên môn của đề tài luận án, là giảng viên hoặc nghiên cứu viên cơ hữu của Nhà trường; 01 người hướng dẫn nghiên cứu sinh có thể tham gia Hội đồng với tư cách là ủy viên;\n\ne) Cha, mẹ, vợ hoặc chồng, con, anh chị em ruột của nghiên cứu sinh không tham gia Hội đồng đánh giá luận án cấp trường.\n\nf) Quy định chi tiết về các yêu cầu, nhiệm vụ và điều kiện đối với từng chức danh trong Hội đồng đánh giá luận án cấp trường như sau:\n\n- Chủ tịch Hội đồng là người có năng lực và uy tín chuyên môn, am hiểu lĩnh vực nghiên cứu của đề tài luận án; có kinh nghiệm trong đào tạo sau đại học và trong chỉ đạo, điều khiển các buổi bảo vệ luận án; chịu trách nhiệm về các hồ sơ liên quan đến việc bảo vệ cấp trường của nghiên cứu sinh; Chủ tịch Hội đồng chỉ đạo hoàn thành các thủ tục liên quan đến buổi bảo vệ;\n\n- Thư ký Hội đồng có nhiệm vụ kiểm tra và chịu trách nhiệm về các hồ sơ của nghiên cứu sinh, hồ sơ liên quan đến buổi bảo vệ, các văn bản nhận xét, tổng hợp các ý kiến nhận xét gửi đến trước buổi bảo vệ, ghi biên bản chi tiết của buổi bảo vệ và hoàn thành các thủ tục liên quan đến buổi bảo vệ để nộp cho Nhà trường;\n\n- Các phản biện phải là những người am hiểu sâu, có uy tín chuyên môn cao trong lĩnh vực khoa học đó. Người phản biện phải có trách nhiệm cao trong đánh giá chất lượng khoa học của luận án; đọc và viết nhận xét trong đó cần ghi rõ luận án có đáp ứng được yêu cầu của một luận án tiến sĩ hay không. Thời gian đọc và gửi nhận xét không quá 01 tháng. Nếu vì lý do nào đó không thể nhận xét được theo đúng thời gian quy định thì phải báo cáo lại để Chủ tịch Hội đồng trình Hiệu trưởng kéo dài thời gian đọc và gửi nhận xét hoặc thay đổi người phản biện nếu cần;\n\n- Các thành viên Hội đồng phải đọc luận án, viết nhận xét và gửi về phòng Đào tạo trước ít nhất 05 ngày trước khi họp Hội đồng đánh giá luận án;\n\n- Để chuẩn bị cho buổi bảo vệ luận án cấp trường, tất cả các thành viên hội đồng phải chuẩn bị các câu hỏi để đánh giá luận án và trình độ của nghiên cứu sinh và sẽ nêu ra ở buổi bảo vệ.\n\n5. Quy trình tổ chức đánh giá luận án cấp Trường"
    ]
    
    chunker = ClusterSemanticChunker()
    
    chunks = chunker.split_text(sentences[0])
    print(chunks)
    print(len(chunks))